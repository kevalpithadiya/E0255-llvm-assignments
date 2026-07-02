===============================================================================
===============================================================================
                    Hoisting Anticipated Expressions in LLVM

                   Author: kevalpithadiya (kevalp@iisc.ac.in)
                    For E0-255 Compiler Design (Spring 2026)
===============================================================================
===============================================================================

*******************************************************************************
                               Results and Caveats
*******************************************************************************

- The pass implementation passes all the provided test cases.

- It takes into account instructions which are commutative by themselves.

- It does not take into account instruction flags (nsw, fast, etc.)

*******************************************************************************
                             Implementation Details
*******************************************************************************

OVERVIEW

The pass is implemented as two nested fixed-point iterations. The inner part is
for computing the anticipated expressions at each block in the CFG using the
standard data-flow analysis technique discussed in class. The outer part is for
iteratively hoisting the anticipated expressions, which is necessary since
hoisting one set of anticipated expressions may result in other expressions
becoming anticipated.

ANTICIPATED EXPRESSIONS ANALYSIS

The data-flow analysis is centered around the notion of expressional equality
between two LLVM instructions and efficient lookup of equal instructions using
hashing. This was accomplished by using a DenseMap data structure which maps
the "expression" computed by an instruction to a set of equivalent instructions
with the help of custom hash and equality functions (named ExprInstrMap).

Two instructions are equal expressionally if and only if they:
1. have the same opcodes,
2. are deterministic given the operands,
3. have the same types,
4. have the same number of operands,
5. have identical operands (respecting order depending on commutativity)
6. have the same attributes (determined by Instruction::hasSameSpecialState)

The set of deterministic instructions is conservatively defined by the pass to
include only unary, binary, cast, comparison operations and call instructions
to functions which do no read or write from memory.

The hash function is defined by combining hash values of the opcode, type, and
operands. If the instruction is commutative, operand hashes are combined in a
canonical order. It does not take into account attributes but this does not
affect the correctness since the equality function requires it and DenseMap is
able to handle hash collisions.

Instruction flags were not taken into account because the LLVM source describes
them as optional data which can be dropped since they do not affect the
"conservative interpretation" of the expression.

Since the data-flow analysis is backward, IN and OUT for basic blocks are
updated in the reverse order to avoid spurious computations and observe faster
convergence.

CODE HOISTING

After the anticipated expressions are computed, the CFG is traversed using BFS.
At any basic block for any expression, if the OUT set contains more than one
instruction, it is hoisted to the current block (if not already present) and
all uses of the other equivalent instructions are replaced by the hoisted one.

BFS traversal guarantees that when we first see that an expression can be
hoisted at a basic block, it will be the earliest possible point where it can
be hoisted. The actual hoisting is done by moving one of the anticipated
instructions just before the basic block terminator.

A set of hoisted instructions and that of un-needed instructions is maintained
by updating them each time an expression is hoisted. The former is used to
ensure that once an instruction is hoisted, it or its anticipated siblings are
never considered for hoisting during further traversal of the CFG. After all
possible hoists have been made, the latter is used to erase the un-needed
instructions from the CFG.

*******************************************************************************
                              AI Usage Disclosure
*******************************************************************************

GitHub Copilot's auto-complete feature was utilized to speedup the development
process. It was used primarily to quickly generate programming constructs which
deal with the complex LLVM classes.

LLMs were utilized to understand how the different APIs of LLVM codebase should
be utilized. This was very helpful since the documentation for the LLVM API is
lacks helpful examples illustrating the intended usage pattern. The prompts are
listed below:

1. Explain what the FileCheck directives in the given file
   (`hoist-anticipated-expressions.ll`) mean.

2. Tell me about the `CallBase::onlyReadsMemory` function in LLVM.

3. What are the semantics of the `readnone` flag in an LLVM function?

4. Answer concisely: How do I check if an LLVM Instruction is a definition? That
   is, it has a result.

    1. What is the behavior of `getName()` on an instruction without result?

    2. How should I define a hash function for instructions? It should take into
       account the opcode, attributes and operands.

    3. What are the operands for a call instruction?

5. What is the difference between `SmallSet` and `SmallPtrSet` in LLVM?

    1. Can I use `SmallPtrSet` for storing integers?

    2. Can I provide a custom hashing function for `SmallSet`?

6. Explain `DenseMapPair` in LLVM.

    1. What is `DenseMapInfo`?

    2. What are the sentinel values used for `Value`s?

    3. How does this translate to `DenseSets`?

    4. Can these data structures safely handle hash collisions?

7. How do I use LLVM lit to run an optimizer test specified in a .ll file
   (FileCheck)

8. What are the requirements for the isEqual function to be implemented for
   DenseMapInfo?

9. LLVM Instruction.getName() not working

    1. Can I retrieve the numbered temporary name as a StringRef?

    2. Is it necessary to give a name to a cloned instruction in for
       inserting it into a basic block?

    3. After doing I.replaceAllUsersWith(X), when I call I.removeFromParent(),
       I get the following error.

       Use still stuck around after Def is destroyed: <badref> = mul i32 %0, %0

    4. Does replaceAllUserWith update users in phi nodes?

Additionally, LLMs were utilized to quickly generate test cases to verify pass
behavior for commutative instructions and instructions with flags.
