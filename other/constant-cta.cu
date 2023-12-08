#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define CTA 3000 // If we assume 2 threads per core, then 656 CTAs are enough to saturate
#define tile 4
#define threads 32
#define logthreads 5
#define widthFactor 4
#define overhead 3 * 1024 + threads + 32
// make threads a power of 2

void inclusive_scan(int *arr, int n, int *result);
__global__ void upsweep(int *arr, int *d_n, int *d_ws);
__global__ void downsweep(int *arr, int *d_n, int *d_ws);
__global__ void kogge_stone(int *n, int *ws, int *offset_access);
__global__ void add_constant(int *start, int *ws, int *partialSumStart);
void printArr(int *arr, int n);
__device__ void printAr(int *arr, int n);
int checkArr(int *source, int *answer, int n);
void highlightArr(int *source, int *answer, int n, int max);

void test_add_constant();

void print_d_arr(int *arr, int start, int end);
void print_line();
__global__ void verify_ws_update(int *d_arr, int *d_ws, int *d_n);
__global__ void print_d(int *arr, int *start, int *end);

int main(int args, char *argv[])
{
    int *arr;
    int *result;
    int n;
    if (args >= 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        n = 1024;
    }
    arr = (int *)malloc(n * sizeof(int));
    result = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = 1;
    }

    inclusive_scan(arr, n, result);

    checkArr(arr, result, n);
    if (args == 3)
        highlightArr(arr, result, n, atoi(argv[2]));

    free(arr);
    free(result);
}

void inclusive_scan(int *arr, int n, int *result)
{
    int *d_arr;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    ////////////////// TIMER START
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    //////////////////////////

    int *d_ws;
    int *d_n;
    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    int ws_num_ints = 1 + CTA + 2 * CTA / (threads * widthFactor); // Equal to 1 + CTA + CTA / (threads * widthFactor) + CTA / (threads * widthFactor)^2 + ...
    cudaMalloc((void **)&d_ws, ws_num_ints * sizeof(int));
    printf("ws_num_ints: %d\n", ws_num_ints);
    int f = 0;
    cudaMemcpy(d_ws, &f, sizeof(int), cudaMemcpyHostToDevice);
    d_ws++;

    size_t block_workspace = (n / CTA + threads + 32 + overhead) * sizeof(int); // this is more than we need. just being conservative, but should adjust later !!!!IMPORTANT
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No issues with setup. block_workspace = %lu\n", block_workspace);
    }
    upsweep<<<CTA, threads, block_workspace>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No issues with upsweep\n");
    }
    

    int *limit;
    int *offset_access;
    int arr_k[3];
    int arr_f[3];
    int to_write = 0;
    cudaMalloc((void **)&limit, sizeof(int));
    cudaMalloc((void **)&offset_access, sizeof(int));
    int j = 10; // arbitrary, really
    int k = CTA;
#pragma unroll
    for (int i = CTA; i > 0; i = j)
    {
        if (j <= 1)
            break;
        j = (i + threads * widthFactor - 1) / (threads * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        kogge_stone<<<j, threads>>>(limit, d_ws, offset_access);
        cudaDeviceSynchronize();
        // printf("kogge_stone<<<%d, %d>>>(%d, d_ws, %d)\n", j, threads, k, f);
        arr_k[to_write] = k;
        arr_f[to_write] = f;
        to_write++;
        f = k;
        k += j;
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No issues with Kogge-Stone\n");
    }
    
// unroll the tree, and don't forget to do it in reverse order
#pragma unroll
    for (int i = to_write - 1; i >= 0; i--)
    {
        k = arr_k[i];
        f = arr_f[i];
        j = (k - f + threads * widthFactor - 1) / (threads * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        add_constant<<<j, threads>>>(limit, d_ws, offset_access);
        // printf("add_constant<<<%d, %d>>>(%d, d_ws, %d)\n", j, threads, k, f);
        cudaDeviceSynchronize();
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No issues with add_constant\n");
    }

    downsweep<<<CTA, threads, block_workspace>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("No issues with downsweep\n");
    }

    /////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ///////////////////////// TIMER END
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("(%d, %f)\n", n, milliseconds);
    cudaMemcpy(result, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

__global__ void upsweep(int *arr, int *d_n, int *ws)
{
    extern __shared__ int sharedArray[];
    __shared__ int tile_sums[threads];            // 256
    __shared__ int shmem_reduction_tiles[32]; // 32
    int n = *d_n;
    int num_to_process_all_blocks = n / CTA;
    int num_to_process_block = n / CTA; // 20992 / 10496 = 2
    if (blockIdx.x == CTA - 1)
    {
        num_to_process_block = n - ((CTA - 1) * num_to_process_block);
    }
    for (int i = 0; i < (num_to_process_block + threads - 1) / threads; i++)
    {
        if (threadIdx.x + i * threads < num_to_process_block)
        {
            sharedArray[threadIdx.x + i * threads] = arr[blockIdx.x * num_to_process_all_blocks + threadIdx.x + i * threads];
        }
    }
    int num_to_process_threads = num_to_process_block / threads;
    int start_for_threads = threadIdx.x * num_to_process_threads;
    if (threadIdx.x == threads - 1)
    {
        num_to_process_threads = num_to_process_block - (threads - 1) * num_to_process_threads;
    }

    /* This computation will be done later
    for (int i = start_for_threads + 1; i < start_for_threads + num_to_process_threads; i++)
    {
        sharedArray[i] = sharedArray[i - 1];
    }
    */
    int partial_sum = 0;
    // if (start_for_threads < 0 || start_for_threads + num_to_process_threads > num_to_process_block)
    //     printf("start = %d, end = %d, size = %d\n", start_for_threads, start_for_threads + num_to_process_threads, num_to_process_block);

    __syncthreads();
    for (int i = start_for_threads; i < start_for_threads + num_to_process_threads; i++)
    {
        partial_sum += sharedArray[i]; // check for bank conflicts
    }
    // printf("%d, %d\n", num_to_process_threads, partial_sum);

    tile_sums[threadIdx.x] = partial_sum; // check for bank conflicts
    __syncthreads();
    int second_partial_sum = 0;
    if (threadIdx.x < 32)
    {
        int i = threadIdx.x;
        do // would do i = 0, but results in bank conflicts. So we altered it.
        {
            if (threadIdx.x * 32 + i < threads) // bounded by number of threads
                second_partial_sum += tile_sums[threadIdx.x * 32 + i];
            i = (i + 1) % 32;
        } while (i != threadIdx.x);
        shmem_reduction_tiles[threadIdx.x] = second_partial_sum;
    }
    __syncthreads();
    // Kogge-Stone
    int temp;
    for (int j = 1; j < 32; j *= 2)
    {
        if ((int)threadIdx.x - j >= 0 && threadIdx.x < 32)
        {
            temp = shmem_reduction_tiles[threadIdx.x - j];
        }
        __syncthreads(); // should ultimately remove because they operate in lockstep. but idk why its not working when removed
        if ((int)threadIdx.x - j >= 0 && threadIdx.x < 32)
        {
            shmem_reduction_tiles[threadIdx.x] += temp;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        ws[blockIdx.x] = shmem_reduction_tiles[31];
}

__global__ void downsweep(int *arr, int *d_n, int *ws) // some access of shared memory has bank conflicts
{
    extern __shared__ int sharedArray[];
    __shared__ int tile_sums[threads];            // 256
    __shared__ int shmem_reduction_tiles[32]; // 32
    int n = *d_n;
    int num_to_process_block = n / CTA; // 20992 / 10496 = 2
    int start_for_blocks = blockIdx.x * num_to_process_block;
    if (blockIdx.x == CTA - 1)
    {
        num_to_process_block = n - ((CTA - 1) * num_to_process_block);
    }
    for (int i = 0; i < (num_to_process_block + threads - 1) / threads; i++)
    {
        if (threadIdx.x + i * threads < num_to_process_block)
        {
            sharedArray[threadIdx.x + i * threads] = arr[start_for_blocks + threadIdx.x + i * threads];
        }
    }
    int num_to_process_threads = num_to_process_block / threads;
    int start_for_threads = threadIdx.x * num_to_process_threads;
    if (threadIdx.x == threads - 1)
    {
        num_to_process_threads = num_to_process_block - (threads - 1) * num_to_process_threads;
    }

    /* This computation will be done later
    for (int i = start_for_threads + 1; i < start_for_threads + num_to_process_threads; i++)
    {
        sharedArray[i] = sharedArray[i - 1];
    }
    */
    int partial_sum = 0;

    __syncthreads();
    
    // NO RACE CONDITIONS HERE
    for (int i = start_for_threads; i < start_for_threads + num_to_process_threads; i++)
    {
        partial_sum += sharedArray[i]; // check for bank conflicts
        sharedArray[i] = partial_sum;
    }
    // printf("%d, %d\n", num_to_process_threads, partial_sum);

    tile_sums[threadIdx.x] = partial_sum; // check for bank conflicts; tile_sums bounded by threads
    __syncthreads();
    int second_partial_sum = 0;
    if (threadIdx.x < 32)
    {
        int i = threadIdx.x;
        do // would do i = 0, but results in bank conflicts. So we altered it.
        {
            if (threadIdx.x * 32 + i < threads)
            {
                second_partial_sum += tile_sums[threadIdx.x * 32 + i];
                tile_sums[threadIdx.x * 32 + i] = second_partial_sum;
            }
            i = (i + 1) % 32;
        } while (i != threadIdx.x);
        shmem_reduction_tiles[threadIdx.x] = second_partial_sum;
    }
    __syncthreads();
    // Kogge-Stone
    int temp;
    for (int j = 1; j < 32; j *= 2)
    {
        if ((int)threadIdx.x - j >= 0 && threadIdx.x < 32)
        {
            temp = shmem_reduction_tiles[threadIdx.x - j];
        }
        __syncthreads(); // should ultimately remove because they operate in lockstep. but idk why its not working when removed
        if ((int)threadIdx.x - j >= 0 && threadIdx.x < 32)
        {
            shmem_reduction_tiles[threadIdx.x] += temp;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32 && threadIdx.x > 0)
    {
        int i = threadIdx.x;
        do // would do for (i = 0; i < 32; i++), but results in bank conflicts. So altered it to use modulo (work on diagonals).
        {
            if (threadIdx.x * 32 + i < threads) // ADJUST num_to_process_block since it doesn't represent shared mem size
                tile_sums[threadIdx.x * 32 + i] += shmem_reduction_tiles[threadIdx.x];
            i = (i + 1) % 32;
        } while (i != threadIdx.x);
    }
    __syncthreads();
    if (threadIdx.x > 0)
    {
        for (int i = start_for_threads; i < start_for_threads + num_to_process_threads; i++)
        {
            sharedArray[i] += tile_sums[threadIdx.x - 1];
        }
    }
    int kernel_level_add = (blockIdx.x > 0) ? ws[blockIdx.x - 1] : 0; // Why is this necessary? I thought I set ws[blockIdx.x - 1] to 0.
    __syncthreads();
    for (int i = 0; i < (num_to_process_block + threads - 1) / threads; i++)
    {
        if (threadIdx.x + i * threads < num_to_process_block)
        {
            arr[start_for_blocks + threadIdx.x + i * threads] = sharedArray[threadIdx.x + i * threads] + kernel_level_add;
        }
    }
}

__global__ void kogge_stone(int *write_start, int *ws, int *offset_access)
{
    __shared__ int arr[threads * widthFactor];
    int offset = *offset_access;
    int limit = *write_start;
    for (int i = 0; i < widthFactor * threads; i += threads)
    {
        if (blockIdx.x * threads * widthFactor + i + threadIdx.x + offset < limit)
        {
            arr[i + threadIdx.x] = ws[blockIdx.x * threads * widthFactor + i + threadIdx.x + offset];
        }
    }
    __syncthreads();
    // kogge-stone computation for separate tiles of size "threads"

    int temp;
    for (int i = 0; i < widthFactor * threads; i += threads)
    {
        for (int j = 1; j < threads; j *= 2)
        {
            if ((int)threadIdx.x - j >= 0)
            {
                temp = arr[threadIdx.x + i - j];
            }
            __syncthreads();
            if ((int)threadIdx.x - j >= 0)
            {
                arr[threadIdx.x + i] += temp;
            }
            __syncthreads();
        }
    }
    // adding the constant to eliminate separation
    for (int i = threads; i < widthFactor * threads; i += threads)
    {
        arr[threadIdx.x + i] += arr[i - 1];
        __syncthreads(); // necessary, since the previous set of threads is updated, the i-1 needs updating first
    }

    // write back to global memory
    for (int i = 0; i < widthFactor * threads; i += threads)
    {
        if (blockIdx.x * threads * widthFactor + i + threadIdx.x + offset < limit)
        {
            ws[blockIdx.x * threads * widthFactor + i + threadIdx.x + offset] = arr[i + threadIdx.x];
        }
    }
    __syncthreads(); // we can find an index such that this is not necessary, DO NOW

    // write back partial sum to next 4-byte integer of global memory
    if (threadIdx.x == 0)
    {
        if (blockIdx.x * threads * widthFactor + threads * widthFactor + offset <= limit)
        {
            ws[limit + blockIdx.x] = arr[threads * widthFactor - 1];
        }
        else
        {
            ws[limit + blockIdx.x] = arr[limit - offset - blockIdx.x * threads * widthFactor - 1];
        }
    }
}

__global__ void add_constant(int *limit, int *ws, int *start)
{
    int start_index = *start + blockIdx.x * threads * widthFactor;
    int lim = *limit;
    int toAdd = (blockIdx.x == 0) ? 0 : *(ws + *limit + blockIdx.x - 1);
    for (int i = 0; i < threads * widthFactor; i += threads)
    {
        if (start_index + i + threadIdx.x < lim)
        {
            ws[start_index + i + threadIdx.x] += toAdd;
        }
        else
        {
            break;
        }
    }
}

__device__ void printAr(int *arr, int n)
{
    printf("[%d", arr[0]);
    for (int i = 1; i < n; i++)
    {
        printf(", %d", arr[i]);
    }
    printf("]\n");
}

void printArr(int *arr, int n)
{
    printf("[%d", arr[0]);
    for (int i = 1; i < n; i++)
    {
        printf(", %d", arr[i]);
    }
    printf("]\n");
}

void print_d_arr(int *arr, int start, int end)
{
    int *d_start;
    int *d_end;
    cudaMalloc((void **)&d_start, sizeof(int));
    cudaMalloc((void **)&d_end, sizeof(int));
    cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, &end, sizeof(int), cudaMemcpyHostToDevice);
    print_d<<<1, 1>>>(arr, d_start, d_end);
    cudaDeviceSynchronize();
}

__global__ void print_d(int *arr, int *start, int *end)
{
    printf("[%d", *(arr + *start));
    for (int i = *start + 1; i < *end; i++)
    {
        printf(", %d", arr[i]);
    }
    printf("]\n");
}

int checkArr(int *source, int *answer, int n)
{
    int total = 0;
    int returnValue = -1;
    for (int i = 0; i < n; i++)
    {
        total += *(source + i);
        if (total != *(answer + i))
        {
            returnValue = i;
            break;
        }
    }
    if (returnValue != -1)
    {
        printf("WRONG! At %d = %d (mod 1024), should be %d, got %d\n[", returnValue, returnValue % 1024, total, *(answer + returnValue));
        for (int i = -10; i < 10; i++)
        {
            printf("%d, ", *(answer + returnValue + i));
        }
        printf("]\n");
    }
    else
    {
        printf("CORRECT");
    }
    return returnValue;
}

void highlightArr(int *source, int *answer, int n, int max)
{
    int total = 0;
    printf("\033[1;32m[");
    for (int i = 0; i < n; i++)
    {
        if (i == max)
            return;
        total += *(source + i);

        if (total == *(answer + i))
        {
            printf("\033[1;32m%d, ", *(answer + i));
        }
        else
        {
            printf("\033[1;31m%d, ", *(answer + i));
        }
    }
    printf("\033[1;32m]\n");
}

void print_line()
{
    printf("\n\n\n");
    for (int i = 0; i < 50; i++)
    {
        printf("-");
    }
    printf("\n\n\n");
}

__global__ void verify_ws_update(int *d_arr, int *d_ws, int *d_n)
{
    int total = 0;
    int n = *d_n;
    for (int i = 0; i < n; i++)
    {
        total += d_arr[i];
        if ((i + 1) % (threads * widthFactor) == 0)
        {
            if (total != d_ws[i / (threads * widthFactor)])
            {
                printf("ws is wrong at index %d, should be %d, but got %d\n", i / (threads * widthFactor), total, d_ws[i / (threads * widthFactor)]);
                return;
            }
        }
    }
    printf("workspace is correct\n");
}

void test_add_constant()
{
    int *arr = (int *)malloc(1075000000 * sizeof(int));
    int *copy = (int *)malloc(1075000000 * sizeof(int));
    for (int i = 0; i < 1073741824; i++)
    {
        arr[i] = 1;
        copy[i] = 1 + (i / 1024) * 1024;
    }
    for (int i = 0; i < 1048600; i++)
    {
        arr[1073741824 + i] = 1024 * (1 + i);
    }
    int limit = 1073741824;
    int *lim;
    cudaMalloc((void **)&lim, sizeof(int));
    cudaMemcpy(lim, &limit, sizeof(int), cudaMemcpyHostToDevice);
    int start = 0;
    int *s;
    cudaMalloc((void **)&s, sizeof(int));
    cudaMemcpy(s, &start, sizeof(int), cudaMemcpyHostToDevice);

    int *d_arr;
    cudaMalloc((void **)&d_arr, 1075000000 * sizeof(int));
    cudaMemcpy(d_arr, arr, 1075000000 * sizeof(int), cudaMemcpyHostToDevice);
    add_constant<<<1048576, threads>>>(lim, d_arr, s);
    cudaDeviceSynchronize();
    cudaMemcpy(arr, d_arr, 1075000000 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 1073741824; i++)
    {
        if (arr[i] != copy[i])
        {
            printf("Messed up at arr[%d]\narr[]:  [", i);
            for (int j = 0 - 10; j < 10; j++)
            {
                printf("%4d, ", arr[i - j]);
            }
            printf("]\ncopy[]: [");
            for (int j = 0 - 10; j < 10; j++)
            {
                printf("%4d, ", copy[i + j]);
            }
            printf("]\n");
            return;
        }
    }
    printf("Successful test on add_constant\n");
}