===============================================================================
===============================================================================
            MLIR Affine Loop Interchange Pass based on Cost Function

                   Author: kevalpithadiya (kevalp@iisc.ac.in)
                    For E0-255 Compiler Design (Spring 2026)
===============================================================================
===============================================================================

*******************************************************************************
                               Results and Caveats
*******************************************************************************

- The pass implementation passes all the provided test cases.

- Currently the cost is stored as uint64_t. While computing total number of
  cache misses for deep nests, it is entirely possible that the cost value can
  overflow, leading to incorrect results. This can be addressed by using the
  APInt datatype which allows for arbitrary bit-width or by implementing
  cost scaling when it gets close to overflowing.

- While writing the pass, I overlooked the case where rectangular loops can
  have variables as loop bounds. The current implementation only considers loop
  bounds which are constants and bails out in other cases.

*******************************************************************************
                             Implementation Details
*******************************************************************************

==> COST FUNCTION

The cost function for each permutation is defined as follows:

    netCost = cacheMissCost / (outerParallelismDegree + 1)

All valid permutations are checked and the one with lowest net cost is applied.

==> CACHE MISS COST

The cache miss cost is computed as the sum of the total number of cache misses
observed by each load / store operation based on temporal and spatial locality.

==> LOOP DISTRIBUTION

Before searching for candidate perfect loop nests, the pass tries to greedily
and recursively distribute imperfect loop nests of the below form, i.e., a loop
which contains only for loops as children.

affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 1024 {...}
    affine.for %j = 1024 to 2048 {...}
    affine.for %j = 2048 to 3072 {...}
}

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

1. How to add a new transformation pass within the affine dialect to MLIR?

2. What are the conditions for validity of loop fission (loop distribution)?

   - Give me a necessary and sufficient condition on the dependences for the
     validity of loop fission. Cite relevant sources.

   - What are hyper-rectangular loop nests?

3. In the MLIR affine dialect, there is a function in AffineAnalysis.cpp
   mlir::affine::checkMemrefAccessDependence. Explain in detail what this
   function does. Refer to files in the 21.1.0 tag in the repository.

   - What are the semantics of the loopDepth argument? As in, how do the
     ordering constraints applied by the function map to actual loop variables?

4. Are there any helper functions in the MLIR affine dialect to check if a loop
   carries a dependence component?

5. How to perform early return in a recursive walk in MLIR?

   - What happens if the IR structure changes during the walk?

   - Explain the general idea and workflow of using PatternRewriter.

6. Explain and give an example use for the NestedMatcher in MLIR Affine
   Analysis.
