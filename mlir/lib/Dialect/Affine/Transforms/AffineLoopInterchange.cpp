//===- AffineLoopInterchange.cpp - Code to perform loop interchange--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop interchange based on a cost model to minimize the
// number of cache misses and maximize parallelism.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <stdlib.h>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPINTERCHANGE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-interchange"

using namespace mlir;
using namespace mlir::affine;

/// Cache line size in bytes.
const uint64_t cacheLineSize = 64;

/// Returns the number of iterations of a rectangular AffineForOp
static inline uint64_t getNumIterations(AffineForOp &forOp) {
  auto lb = forOp.getConstantLowerBound();
  auto ub = forOp.getConstantUpperBound();
  auto step = forOp.getStepAsInt();
  return (ub - lb + step - 1) / step;
}

/// Computes the cost of a loop permutation based on the expected number of
/// cache misses for the load and store operations.
static uint64_t
getPermutationCacheCost(const ArrayRef<AffineForOp> &nestedLoops,
                        const ArrayRef<Operation *> &loadAndStoreOps,
                        const ArrayRef<unsigned> &permMap) {
  // This function should only be invoked when there are multiple loops.
  assert(nestedLoops.size() > 1 &&
         "Expected multiple loops for permutation cost computation");

  // Accumulate the total cost quantified as the sum of total number of cache
  // misses for each load / store operation for the entire loop nest.
  uint64_t totalCost = 0;

  // Obtain inverted permutation map
  SmallVector<unsigned, 2> invertedPermMap(permMap.size());
  for (unsigned i = 0; i < permMap.size(); ++i) {
    invertedPermMap[permMap[i]] = i;
  }

  // Compute cost for each load / store operation
  for (Operation *op : loadAndStoreOps) {
    MemRefType memRefType;
    AffineValueMap affineValueMap;
    uint64_t operationCost;

    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      memRefType = loadOp.getMemref().getType();
      affineValueMap =
          AffineValueMap(loadOp.getAffineMap(), loadOp.getIndices());
      operationCost = 1;
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      memRefType = storeOp.getMemref().getType();
      affineValueMap =
          AffineValueMap(storeOp.getAffineMap(), storeOp.getIndices());
      operationCost = 2;
    } else {
      assert(false && "Expected only AffineLoadOp or AffineStoreOp");
    }

    affineValueMap.composeSimplifyAndCanonicalize();

    // Compute innermost loop cost
    auto innermostLoop = nestedLoops[invertedPermMap.back()];
    auto innermostLoopIV = innermostLoop.getInductionVar();
    bool innermostTemporalReuse = false;

    // If the non-innermost access dimension is a function of the innermost
    // loop IV, then there will be cache miss in each iteration.
    if (affineValueMap.isFunctionOf(0u, affineValueMap.getNumResults() - 1,
                                    innermostLoopIV)) {
      operationCost *= getNumIterations(innermostLoop);
    }
    // If only the innermost access dimension is a function of the
    // innermost loop IV, spacial locality may be observed. Hence number
    // of cache misses will be the number of iterations divided by the
    // number of accessed elements that fit in a cache line. This requires
    // consideration of the access stride and size of the accessed element.
    else if (affineValueMap.isFunctionOf(affineValueMap.getNumResults() - 1,
                                         innermostLoopIV)) {
      // Perform spatial locality based cost only if the layout is row-major.
      // TODO: Take into access row-major strided layouts as well.
      if (memRefType.getLayout().isIdentity()) {
        auto elementSize =
            memRefType.getElementType().getIntOrFloatBitWidth() / 8;
        // TODO: Also take into account the access stride which is the
        // coefficient of the innermost loop IV in the innermost access
        // dimension.
        auto accessStride = 1;
        auto elementsPerCacheLine =
            cacheLineSize / (elementSize * accessStride);
        operationCost *=
            (getNumIterations(innermostLoop) + elementsPerCacheLine - 1) /
            elementsPerCacheLine;
      }
      // Otherwise, conservatively assume no spatial locality.
      else {
        operationCost *= getNumIterations(innermostLoop);
      }
    }
    // If the access is not a function of the innermost loop IV, there is only
    // one cache miss (temporal reuse) in the innermost loop and the cost is
    // unchanged.
    else {
      innermostTemporalReuse = true;
    }

    // Consider the remaining loops from innermost to outermost for temporal
    // reuse. It can only be observed for a maximal contiguous set of loops
    // starting from the innermost loop. Once a loop is encountered that does
    // not show temporal reuse, further outer loops cannot show temporal reuse.
    for (int i = permMap.size() - 2; i >= 0; --i) {
      auto loop = nestedLoops[invertedPermMap[i]];
      if (innermostTemporalReuse) {
        auto loopIV = loop.getInductionVar();
        if (affineValueMap.isFunctionOf(0u, affineValueMap.getNumResults() - 1,
                                        loopIV)) {
          operationCost *= getNumIterations(loop);
          innermostTemporalReuse = false;
        }
      } else {
        operationCost *= getNumIterations(loop);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Cost for operation at location: ";
               op->getLoc().dump();
               llvm::dbgs() << " is: " << operationCost << "\n";);

    totalCost += operationCost;
  }

  return totalCost;
}

static uint64_t getPermutationParallelism(
    const std::vector<SmallVector<DependenceComponent, 2>> &depCompsVec,
    const ArrayRef<unsigned> &permMap) {
  // Obtain inverted permutation map
  SmallVector<unsigned, 2> invertedPermMap(permMap.size());
  for (unsigned i = 0; i < permMap.size(); ++i) {
    invertedPermMap[permMap[i]] = i;
  }

  // Obtain outer parallelism degree by scanning the dependence compoenents in
  // the order of the permuted loops and counting the number of loops which do
  // no carry the dependence.
  uint64_t parallelismDegree = 0;
  for (unsigned i = 0; i < permMap.size(); ++i) {
    bool isParallel = true;

    for (const auto &depComps : depCompsVec) {
      auto ub = depComps[invertedPermMap[i]].ub;
      // If the dependence is unbounded above or has a positive upper bound,
      // then the current loop carries the dependence and is not parallel.
      if (!ub.has_value() || ub.value() > 0) {
        isParallel = false;
        break;
      }
    }

    // If current loop is parallel, increase parallelism degree.
    if (isParallel)
      parallelismDegree++;
    // Otherwise, further loops cannot be parallel.
    else
      break;
  }

  return parallelismDegree;
}

static SmallVector<unsigned, 2>
findOptimalLoopPermutation(const ArrayRef<AffineForOp> &nestedLoops,
                           const ArrayRef<Operation *> &loadAndStoreOps) {
  // Initialize permutation map to identity permutation.
  SmallVector<unsigned, 2> permMap(nestedLoops.size());
  std::iota(permMap.begin(), permMap.end(), 0);

  // Obtain the dependence components for the loop nest. Required for
  // checking interchange validity and degree of outer parallelism.
  std::vector<SmallVector<DependenceComponent, 2>> depCompsVec;
  getDependenceComponents(nestedLoops[0], nestedLoops.size(), &depCompsVec);

  // Current optimal permutation and its cost.
  SmallVector<unsigned, 2> optimalPermMap = permMap;
  uint64_t optimalNetCost = std::numeric_limits<uint64_t>::max();

  // Loop over all permutations.
  do {
    LLVM_DEBUG(llvm::dbgs() << "Evaluating permutation: ";
               for (unsigned i : permMap) llvm::dbgs() << i << " ";
               llvm::dbgs() << "\n";);

    if (!isValidLoopInterchangePermutation(nestedLoops, permMap)) {
      LLVM_DEBUG(llvm::dbgs() << "Permutation is invalid with respect to data "
                                 "dependencies, skipping.\n";);
      continue;
    }

    uint64_t cacheCost =
        getPermutationCacheCost(nestedLoops, loadAndStoreOps, permMap);
    uint64_t parallelismDegree =
        getPermutationParallelism(depCompsVec, permMap);

    // Heuristic cost mdoel to incorporate both cache cost and parallelism.
    uint64_t netCost = cacheCost / (parallelismDegree + 1);

    LLVM_DEBUG(llvm::dbgs() << "Cache: " << cacheCost
                            << ", Parallelism: " << parallelismDegree
                            << ", Net Cost: " << netCost << "\n";);

    if (netCost < optimalNetCost) {
      optimalNetCost = netCost;
      optimalPermMap = permMap;
    }

  } while (std::next_permutation(permMap.begin(), permMap.end()));

  LLVM_DEBUG(llvm::dbgs() << "Optimal loop permutation: ";
             for (unsigned idx : optimalPermMap) llvm::dbgs() << idx << " ";
             llvm::dbgs() << " with cost: " << optimalNetCost << "\n";);

  return optimalPermMap;
}

namespace {

struct AffineLoopInterchangePass
    : public affine::impl::AffineLoopInterchangeBase<
          AffineLoopInterchangePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs()
                   << "\n\nRunning Affine Loop Interchange Pass on function: "
                   << funcOp.getName() << "\n";);

    // Perform loop distribution for imperfect loop nests of 2 levels at a time
    // where the inner loop only has for ops. Distribution of arbitrary loops
    // can require spilling intermediate results.
    bool checkForDistribution = true;
    while (checkForDistribution) {
      checkForDistribution = false;

      // To find an imperfect loop nest, look for a for op which
      // - is an immediate parent for op, and
      // - is the first op in the parent for op's body, and
      // - has one or more sibling for ops and no other sibling non-for ops.
      funcOp.walk([&](AffineForOp forOp) -> WalkResult {
        // Check if the parent op is a for op.
        Operation *parentOp = forOp.getOperation()->getParentOp();
        auto parentForOp = dyn_cast<AffineForOp>(parentOp);
        if (!parentForOp)
          return WalkResult::advance();

        // Parent for op should not have iteration arguments.
        if (!parentForOp.getRegionIterArgs().empty())
          return WalkResult::advance();

        // Check if the for op is the first op in the parent for op's body.
        Block *loopBody = parentForOp.getBody();
        if (forOp.getOperation() != &loopBody->front())
          return WalkResult::advance();

        // Collect sibling ops and check if there is any non-for sibling op.
        SmallVector<AffineForOp, 1> innerForOps{};

        for (Operation &op : loopBody->getOperations()) {
          if (auto siblingForOp = dyn_cast<AffineForOp>(op)) {
            if (siblingForOp != forOp)
              innerForOps.push_back(siblingForOp);
          }
          // If it is not a for op or terminator of parent loop,
          // skip this candidate.
          else if (!isa<AffineYieldOp>(op))
            return WalkResult::advance();
        }

        // If there is no sibling for op, skip this candidate.
        if (innerForOps.size() == 0)
          return WalkResult::advance();

        // Check if the loop nest is distributable using conservative check.
        if (hasCyclicDependence(parentForOp))
          return WalkResult::advance();

        LLVM_DEBUG(llvm::dbgs() << "Distributing imperfect loop nest: \n";
                   llvm::dbgs() << "Parent for op: ";
                   parentForOp.getLoc().dump();
                   llvm::dbgs() << "Child for op: "; forOp.getLoc().dump(););

        // Perform loop distribution by creating a duplicate parent loop and
        // moving the sibling for ops into the body of the duplicated loop.

        // Clone and insert the duplicate parent loop after the original
        // parent loop.
        OpBuilder builder(parentForOp);
        builder.setInsertionPointAfter(parentForOp);
        auto newParentForOp = builder.create<AffineForOp>(
            parentForOp.getLoc(), parentForOp.getLowerBoundOperands(),
            parentForOp.getLowerBoundMap(), parentForOp.getUpperBoundOperands(),
            parentForOp.getUpperBoundMap(), parentForOp.getStepAsInt());

        Value oldIV = parentForOp.getInductionVar();
        Value newIV = newParentForOp.getInductionVar();

        for (auto childForOp : innerForOps) {
          // Move the child for op to the body of the new parent for op.
          childForOp.getOperation()->moveBefore(
              newParentForOp.getBody()->getTerminator());

          // Update uses of the induction variable in the moved operations.
          childForOp.walk(
              [&](Operation *op) { op->replaceUsesOfWith(oldIV, newIV); });
        }

        checkForDistribution = true;
        return WalkResult::interrupt();
      });
    }

    // Look for candidate perfectly nested loop nests for interchange.
    funcOp.walk([&](AffineForOp forOp) {
      // Only walk over perfectly nested for loops by early return if the
      // forOp has a parent forOp.
      Operation *forOpOp = forOp.getOperation();
      if (forOpOp->getParentOfType<AffineForOp>())
        return;

      // Obtain the maximal list of perfectly nested loops.
      SmallVector<AffineForOp> nestedLoops;
      getPerfectlyNestedLoops(nestedLoops, forOp);

      // Skip if there is only one loop in the nest.
      if (nestedLoops.size() == 1)
        return;

      // Skip if loop is not hyper-rectangular with constant bounds.
      for (AffineForOp nestedLoop : nestedLoops) {
        if (!nestedLoop.hasConstantBounds())
          return;
      }

      // Skip interchange if one of the following conditions hold:
      // 1) There is a further nested loop (imperfect nest).
      // 2) There is an affine.if op in the loop body.
      bool skipInterchange = false;
      AffineForOp innermostLoop = nestedLoops.back();
      // Also collect the memory accesses in the innermost in the same walk.
      SmallVector<Operation *, 4> loadAndStoreOps;

      // Walk the innermost loop body to check for above conditions
      innermostLoop.walk([&skipInterchange, &loadAndStoreOps,
                          innermostLoop](Operation *op) mutable -> WalkResult {
        if (auto childForOp = dyn_cast<AffineForOp>(op)) {
          if (childForOp != innermostLoop) {
            skipInterchange = true;
            WalkResult::interrupt();
          }
        }

        if (isa<AffineIfOp>(op)) {
          skipInterchange = true;
          WalkResult::interrupt();
        }

        if (isa<AffineLoadOp, AffineStoreOp>(op)) {
          loadAndStoreOps.push_back(op);
        }

        return WalkResult::advance();
      });
      if (skipInterchange)
        return;

      // Print the perfect loop nest.
      LLVM_DEBUG(llvm::dbgs() << "Found perfect loop nest of depth at ";
                 nestedLoops.front().getLoc().dump(););

      // Find and apply the optimal loop permutation for the loop nest.
      SmallVector<unsigned, 2> optimalPermMap =
          findOptimalLoopPermutation(nestedLoops, loadAndStoreOps);
      permuteLoops(nestedLoops, optimalPermMap);
    });
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopInterchangePass() {
  return std::make_unique<AffineLoopInterchangePass>();
}
