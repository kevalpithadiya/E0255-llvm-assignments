//===-- HoistAnticipatedExpressions.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/HoistAnticipatedExpressions.h"

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "hoist-anticipated-expressions-pass"

bool HoistAnticipatedExpressionsPass::isDeterministic(const Instruction *I) {
  return (I->isUnaryOp() || I->isBinaryOp() || I->isCast() || isa<CmpInst>(I) ||
          (isa<CallInst>(I) && !I->mayReadOrWriteMemory()));
}

Instruction *HoistAnticipatedExpressionsPass::ExpressionInfo::getEmptyKey() {
  uintptr_t Val = static_cast<uintptr_t>(-1);
  Val <<= Log2MaxAlign;
  return reinterpret_cast<Instruction *>(Val);
}

Instruction *
HoistAnticipatedExpressionsPass::ExpressionInfo::getTombstoneKey() {
  uintptr_t Val = static_cast<uintptr_t>(-2);
  Val <<= Log2MaxAlign;
  return reinterpret_cast<Instruction *>(Val);
}

unsigned HoistAnticipatedExpressionsPass::ExpressionInfo::getHashValue(
    const Instruction *I) {
  unsigned OpcodeHash = DenseMapInfo<unsigned>::getHashValue(I->getOpcode());
  unsigned TypeHash = DenseMapInfo<Type *>::getHashValue(I->getType());
  unsigned Hash = detail::combineHashValue(OpcodeHash, TypeHash);

  // If I is commutative, compute hashes of the two operands based on their
  // pointer addresses rather than order of appearance in the instruction
  if (I->isCommutative()) {
    Value *O0 = I->getOperand(0), *O1 = I->getOperand(1);
    unsigned O0Hash = DenseMapInfo<Value *>::getHashValue(O0);
    unsigned O1Hash = DenseMapInfo<Value *>::getHashValue(O1);
    unsigned OperandsHash = (O0 <= O1)
                                ? detail::combineHashValue(O0Hash, O1Hash)
                                : detail::combineHashValue(O1Hash, O0Hash);
    Hash = detail::combineHashValue(Hash, OperandsHash);
  }
  // Otherwise simply compute hash of operands in the order in instruction
  else {
    for (const Value *Operand : I->operands()) {
      Hash = detail::combineHashValue(
          Hash, DenseMapInfo<Value *>::getHashValue(Operand));
    }
  }

  return Hash;
}

bool HoistAnticipatedExpressionsPass::ExpressionInfo::isEqual(
    const Instruction *L, const Instruction *R) {
  if (L == R)
    return true;

  if (L == getEmptyKey() || L == getTombstoneKey() || R == getEmptyKey() ||
      R == getTombstoneKey())
    return false;

  if (L->getOpcode() != R->getOpcode())
    return false;

  // Always return false if the operation is not deterministic
  // (shouldn't depend on memory or have exceptional control flow)
  if (!isDeterministic(L) || !isDeterministic(R))
    return false;

  if (L->getType() != R->getType())
    return false;
  if (L->getNumOperands() != R->getNumOperands())
    return false;

  // If both instructions are commutative, it is a binrary operation.
  // Return false if operands are the same irrespective of order.
  if (L->isCommutative()) {
    auto L0 = L->getOperand(0), L1 = L->getOperand(1);
    auto R0 = R->getOperand(0), R1 = R->getOperand(1);

    if (!((L0 == R0 && L1 == R1) || (L0 == R1 && L1 == R0)))
      return false;
  }
  // Otherwise match all operands respecting their order.
  else {
    for (const auto [LeftOperand, RightOperand] :
         zip(L->operands(), R->operands())) {
      if (LeftOperand != RightOperand)
        return false;
    }
  }

  // Return true only if the instructions also have the same special state.
  // The special state includes
  return L->hasSameSpecialState(R);
}

void HoistAnticipatedExpressionsPass::printExprInstrMap(
    const ExprInstrMap &Map) {
  LLVM_DEBUG(for (const auto &[K, V] : Map) {
    K->printAsOperand(llvm::dbgs(), false);
    llvm::dbgs() << " : ";
    for (const auto *I : V) {
      I->printAsOperand(llvm::dbgs(), false);
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "\n";
  });
}

HoistAnticipatedExpressionsPass::ExprInstrMap
HoistAnticipatedExpressionsPass::setExprUnion(const ExprInstrMap &A,
                                              const ExprInstrMap &B) {
  // Initialize result as a copy of A
  ExprInstrMap Result = ExprInstrMap();

  // Add all entries from A to the result
  for (const auto &[Expr, Instrs] : A) {
    Result[Expr] = SmallPtrSet<Instruction *, 2>(Instrs.begin(), Instrs.end());
  }

  for (const auto &[Expr, Instrs] : B) {
    if (Result.find(Expr) == Result.end()) {
      Result[Expr] =
          SmallPtrSet<Instruction *, 2>(Instrs.begin(), Instrs.end());
    } else {
      Result[Expr].insert(Instrs.begin(), Instrs.end());
    }
  }

  return Result;
}

HoistAnticipatedExpressionsPass::ExprInstrMap
HoistAnticipatedExpressionsPass::setExprIntersection(const ExprInstrMap &A,
                                                     const ExprInstrMap &B) {
  ExprInstrMap Result = ExprInstrMap();

  // If an expression is in both A and B, union both of their instruction sets
  // and add to the result
  for (const auto &[Expr, Instrs] : A) {
    if (B.find(Expr) != B.end()) {
      Result[Expr] =
          SmallPtrSet<Instruction *, 2>(Instrs.begin(), Instrs.end());

      const auto &BInstrs = B.at(Expr);
      Result[Expr].insert(BInstrs.begin(), BInstrs.end());
    }
  }

  return Result;
}

HoistAnticipatedExpressionsPass::ExprInstrMap
HoistAnticipatedExpressionsPass::setExprSubtraction(const ExprInstrMap &A,
                                                    const ExprInstrMap &B) {
  ExprInstrMap Result = ExprInstrMap();

  for (const auto &[Expr, Instrs] : A) {
    // If Expr is in A but not in B, add it to the result with the same
    // instruction set
    if (B.find(Expr) == B.end()) {
      Result[Expr] =
          SmallPtrSet<Instruction *, 2>(Instrs.begin(), Instrs.end());
    } else {
      // If Expr is in both A and B, add it to the result with the difference of
      // their instruction sets
      const auto &BInstrs = B.at(Expr);
      SmallPtrSet<Instruction *, 2> Difference;
      for (Instruction *I : Instrs) {
        if (!BInstrs.count(I)) {
          Difference.insert(I);
        }
      }

      if (!Difference.empty())
        Result[Expr] = Difference;
    }
  }

  return Result;
}

PreservedAnalyses
HoistAnticipatedExpressionsPass::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(llvm::dbgs()
                 << "\n\nRunning HoistAnticipatedExpressionsPass on function: "
                 << F.getName() << "\n";);

  bool Hoisted = true;

  while (Hoisted) {
    Hoisted = false;

    // Allocate Data-flow Analysis Sets
    size_t NumBasicBlocks = F.size();
    DenseMap<const BasicBlock *, ExprInstrMap> VUse(NumBasicBlocks);
    DenseMap<const BasicBlock *, ExprInstrMap> VDef(NumBasicBlocks);
    DenseMap<const BasicBlock *, ExprInstrMap> IN(NumBasicBlocks);
    DenseMap<const BasicBlock *, ExprInstrMap> OUT(NumBasicBlocks);

    ExprInstrMap UniversalSet = ExprInstrMap();

    // Initialize the DFA sets
    for (BasicBlock &BB : F) {
      LLVM_DEBUG(llvm::dbgs() << "\nBasic Block: ";
                 BB.printAsOperand(llvm::dbgs(), false); llvm::dbgs() << "\n";);

      // Initialize entries for the current BasicBlock
      ExprInstrMap CurVUse = ExprInstrMap();
      ExprInstrMap CurVDef = ExprInstrMap();

      for (Instruction &I : BB) {
        // Add current instruction to the gen set if it's operands
        // are not defined in the same BasicBlock and the
        // expression is deterministic.
        if (isDeterministic(&I)) {
          // Add all deterministic instructions to the universal set of
          // expressions
          UniversalSet[&I].insert(&I);

          bool DefInSameBB = false;
          for (const auto &Op : I.operands()) {
            if (const Instruction *Def = dyn_cast<Instruction>(Op)) {
              DefInSameBB |= (Def->getParent() == &BB);
            }
          }

          if (!DefInSameBB) {
            CurVUse[&I].insert(&I);
          }
        }

        // Add all users of current instruction to the kill set
        for (auto *User : I.users()) {
          if (Instruction *UserI = dyn_cast<Instruction>(User)) {
            if (isDeterministic(UserI))
              CurVDef[UserI].insert(UserI);
          }
        }
      }

      VUse[&BB] = CurVUse;
      VDef[&BB] = CurVDef;

      // Print the computed VUse and VDef sets
      LLVM_DEBUG({
        llvm::dbgs() << "VUse size: " << VUse[&BB].size() << "\n";
        printExprInstrMap(VUse[&BB]);
        llvm::dbgs() << "VDef size: " << VDef[&BB].size() << "\n";
        printExprInstrMap(VDef[&BB]);
      });
    }

    // Initialize the IN set to the universal set of expressions and OUT set to
    // empty for all BasicBlocks
    for (const BasicBlock &BB : F) {
      IN[&BB] = UniversalSet;
      OUT[&BB] = ExprInstrMap();
    }

    // Print the initial IN and OUT sets
    LLVM_DEBUG(for (const BasicBlock &BB : F) {
      llvm::dbgs() << "\nBasic Block: ";
      BB.printAsOperand(llvm::dbgs(), false);
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  IN size: " << IN[&BB].size() << "\n";
      printExprInstrMap(IN[&BB]);
      llvm::dbgs() << "  OUT size: " << OUT[&BB].size() << "\n";
      printExprInstrMap(OUT[&BB]);
    });

    // Perform iterative data-flow analysis to compute IN and OUT sets
    bool DFAChanged = true;
    int Iteration = 0;

    while (DFAChanged) {
      DFAChanged = false;
      Iteration += 1;

      LLVM_DEBUG(llvm::dbgs() << "\n === Iteration: " << Iteration << "\n";);

      for (const BasicBlock &BB : llvm::reverse(F)) {
        // Compute OUT[B] as the intersection of IN sets of all successors
        ExprInstrMap NewOUT;
        bool First = true;
        for (const BasicBlock *Succ : successors(&BB)) {
          if (First) {
            NewOUT = IN[Succ];
            First = false;
          } else {
            NewOUT = setExprIntersection(NewOUT, IN[Succ]);
          }
        }

        // Compute IN[B] as VUse[B] union (OUT[B] - VDef[B])
        ExprInstrMap NewIN =
            setExprUnion(VUse[&BB], setExprSubtraction(NewOUT, VDef[&BB]));

        // Print the new IN and OUT sets for the current BasicBlock
        LLVM_DEBUG({
          llvm::dbgs() << "\nBasic Block: ";
          BB.printAsOperand(llvm::dbgs(), false);
          llvm::dbgs() << "\n";
          llvm::dbgs() << "  New IN size: " << NewIN.size() << "\n";
          printExprInstrMap(NewIN);
          llvm::dbgs() << "  New OUT size: " << NewOUT.size() << "\n";
          printExprInstrMap(NewOUT);
        });

        // Check if IN or OUT changed for the current BasicBlock
        if (NewIN != IN[&BB] || NewOUT != OUT[&BB]) {
          DFAChanged = true;
          IN[&BB] = NewIN;
          OUT[&BB] = NewOUT;
        }
      }
    }

    // Print the final IN and OUT sets
    LLVM_DEBUG(for (const BasicBlock &BB : F) {
      llvm::dbgs() << "\nBasic Block: ";
      BB.printAsOperand(llvm::dbgs(), false);
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  Final IN size: " << IN[&BB].size() << "\n";
      printExprInstrMap(IN[&BB]);
      llvm::dbgs() << "  Final OUT size: " << OUT[&BB].size() << "\n";
      printExprInstrMap(OUT[&BB]);
    });

    // Traverse the CFG in breadth-first manner starting from the entry block
    // and hoist anticipated expressions at the earliest block possible
    DenseSet<Instruction *> HoistedExpressions = DenseSet<Instruction *>();
    DenseSet<Instruction *> ToRemove = DenseSet<Instruction *>();

    for (BasicBlock *BB : breadth_first(&F.getEntryBlock())) {
      // For each expression in OUT[BB] with more than one instruction,
      // hoist the expression at the end of current BasicBlock if it is not
      // already present in the BasicBlock.
      for (const auto &[Expr, Instrs] : OUT[BB]) {
        if ((Instrs.size() > 1) && !ToRemove.count(Expr) &&
            !HoistedExpressions.count(Expr)) {
          // Check if the expression is present in the current block
          Instruction *HoistedExpression = nullptr;
          for (Instruction *I : Instrs) {
            if (I->getParent() == BB) {
              HoistedExpression = I;
              break;
            }
          }

          // Otherwise clone and hoist the expression
          if (!HoistedExpression) {
            HoistedExpression = Expr;
            HoistedExpression->moveBefore(BB->getTerminator()->getIterator());
          }

          // Log hoisted expression
          LLVM_DEBUG({
            llvm::dbgs() << "Hoisted expression: ";
            HoistedExpression->print(llvm::dbgs());
            llvm::dbgs() << " to Basic Block: ";
            BB->printAsOperand(llvm::dbgs(), false);
            llvm::dbgs() << "\n";
          });

          // Replace all uses of the anticipated expressions in the program
          // with the anticipated expression. This might result in some phi
          // nodes becoming trivial.
          for (Instruction *I : Instrs) {
            if (I != HoistedExpression) {
              I->replaceAllUsesWith(HoistedExpression);
              ToRemove.insert(I);
            }
          }

          // Add to the set of hoisted expressions to avoid hoisting
          // the same expression again.
          HoistedExpressions.insert(HoistedExpression);
        }
      }
    }

    // Remove all instructions that have been replaced by the hoisted
    // expressions
    for (Instruction *I : ToRemove) {
      I->eraseFromParent();
    }

    Hoisted = !HoistedExpressions.empty();
  }

  return PreservedAnalyses::none();
}
