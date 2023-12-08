# prefix-sum
cub.cu - CUB Library implementation of prefix-sums
thrust.cu - Thrust Library implementation of prefix-sums

The above two CUDA files were used for comparing against library implementations.

src/4n.cu is an implementation that minimizes the GPU computation by not re-doing any computation, but in exchange, requires ~4n global memory accesses

src/redo-computation.cu is an implementation that only uses ~3n global memory accesses. This implementaiton involved little to no optimization, and strictly used binary tree for the first and last kernels (to generate local aggregate sum and local prefix sum, respectively) and used Kogge-Stone method to perform upsweep on the aggregate sums. This was used as a standard for comparison for the successive files.

src/shfl-3n.cu is an implementation that uses the SHFL library, and has variable thread granularity. I tested this particular file with a number of different parameters, including number of warps per block, number of elements processed per thread for upsweep, number of threads processed at the base case, etc.

src/single-warp.cu is an implementation that uses a single warp for each block. The idea behind this is that there would be much less synchronization barriers, and all computation can be done using strictly registers (and a few writes to global memory), since there is no need to communicate with other warps.

src/kogge-and-binary.cu is an implementation that involves both the usage of the Kogge-Stone method and the binary tree to compute prefix sums. This file is the most highly optimized one.
