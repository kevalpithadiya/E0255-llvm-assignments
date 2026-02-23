//===-- HoistAnticipatedExpressions.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_HOISTANTICIPATEDEXPRESSIONS_H
#define LLVM_TRANSFORMS_UTILS_HOISTANTICIPATEDEXPRESSIONS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class HoistAnticipatedExpressionsPass
    : public PassInfoMixin<HoistAnticipatedExpressionsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  // Checks if the instruction can be deterministic given the
  // operands, so that only such instructions are considered
  // for the analysis.
  static bool isDeterministic(const Instruction *I);

  // Defines Instruction hashing and equality for the purpose of
  // anticipated expression analysis and hoisting.
  struct ExpressionInfo {
    static constexpr uintptr_t Log2MaxAlign = 12;

    static inline Instruction *getEmptyKey();
    static inline Instruction *getTombstoneKey();

    // Compute the hash of an instruction based on the
    // opcode, type, operands, and commutativity.
    static unsigned getHashValue(const Instruction *I);

    // Equality of two instructions based on them producing identical results.
    // It takes into account opcode, type, operands, flags and commutativity.
    static bool isEqual(const Instruction *L, const Instruction *R);
  };

  // A mapping of expressions to the set of instructions they appear in.
  using ExprInstrMap =
      DenseMap<Instruction *, SmallPtrSet<Instruction *, 2>, ExpressionInfo>;

  // Set operations for ExprInstrMap
  ExprInstrMap setExprUnion(const ExprInstrMap &A, const ExprInstrMap &B);
  ExprInstrMap setExprIntersection(const ExprInstrMap &A,
                                   const ExprInstrMap &B);
  ExprInstrMap setExprSubtraction(const ExprInstrMap &A, const ExprInstrMap &B);

  // Utility function to print all instructions in an ExprInstrMap
  void printExprInstrMap(const ExprInstrMap &Map);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_HOISTANTICIPATEDEXPRESSIONS_H
