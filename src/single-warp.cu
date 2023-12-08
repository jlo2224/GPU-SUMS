#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define threads 32 // now these are threads only for the global aggregate sums, not the prefix sums
#define logthreads 5
#define widthFactor 16
#define CPUwarpSize 32
#define FULL_MASK 0xffffffff
// make threads a power of 2

void inclusive_scan(int *arr, int n, int *result);
__global__ void prefix_sum(int *arr, int *d_n, int *d_ws);
__global__ void downsweep(int *arr, int *d_n, int *d_ws);
__global__ void kogge_stone(int *n, int *ws, int *offset_access);
__global__ void add_constant(int *start, int *ws, int *partialSumStart);
__global__ void add_global_aggregates(int *arr, int *d_n, int *ws);
__global__ void compute_aggregates(int *arr, int *n, int *ws);
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

    //printArr(arr, 10);
    inclusive_scan(arr, n, result);

    checkArr(arr, result, n);
    if (args == 3)
        highlightArr(arr, result, n, atoi(argv[2]));

    free(arr);
    free(result);
}

void inclusive_scan(int *arr, int n, int *result)
{
    cudaError_t cudaError = cudaGetLastError();
    int *d_arr;
    int *d_ws;
    int ws_size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaMalloc((void **)&d_arr, n * sizeof(int) + 1024 * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    ////////////////// TIMER START
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int *d_n;

    int blocks = (n + CPUwarpSize * widthFactor - 1) / (CPUwarpSize * widthFactor); 
    ws_size = blocks + 2 * ((blocks + CPUwarpSize * widthFactor - 1) / (CPUwarpSize * widthFactor));
    int f = 0;
    cudaMalloc((void **)&d_ws, ws_size * sizeof(int));
    cudaMemcpy(d_ws, &f, sizeof(int), cudaMemcpyHostToDevice);
    d_ws++;
    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    compute_aggregates<<<blocks, CPUwarpSize>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();
    /*
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    else
    {
        printf("No issues with compute_aggregate\n");
    }
    */

    int *limit;
    int *offset_access;
    int arr_k[3];
    int arr_f[3];
    int to_write = 0;
    cudaMalloc((void **)&limit, sizeof(int));
    cudaMalloc((void **)&offset_access, sizeof(int));
    int j = 10; // arbitrary, really
    int k = blocks;
    for (int i = blocks; i > 0; i = j)
    {
        if (j <= 1)
            break;
        j = (i + CPUwarpSize * widthFactor - 1) / (CPUwarpSize * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        kogge_stone<<<j, CPUwarpSize>>>(limit, d_ws, offset_access);
        cudaDeviceSynchronize();
        arr_k[to_write] = k;
        arr_f[to_write] = f;
        to_write++;
        f = k;
        k += j;
    }
    cudaError = cudaGetLastError();
    /*
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    else
    {
        printf("No issues with kogge_stone\n");
    }
    */
    // unroll the tree, and don't forget to do it in reverse order
    for (int i = to_write - 1; i >= 0; i--)
    {
        k = arr_k[i];
        f = arr_f[i];
        j = (k - f + CPUwarpSize * widthFactor - 1) / (CPUwarpSize * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        add_constant<<<j, CPUwarpSize>>>(limit, d_ws, offset_access);
        cudaDeviceSynchronize();
    }
    // verify_ws_update<<<1, 1>>>(d_arr, d_ws, d_n);
    // cudaDeviceSynchronize();
    // print_line();
    /*
    cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    else
    {
        printf("No issues with add_constant\n");
    }
    */

    prefix_sum<<<blocks, CPUwarpSize>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();

    /*
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    else
    {
        printf("No issues with prefix_sum\n");
    }
    */

    cudaFree(d_ws);
    cudaFree(d_n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ///////////////////////// TIMER END
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("(%d, %f)\n", n, milliseconds);
    cudaMemcpy(result, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

__global__ void compute_aggregates(int *arr, int *n, int *ws)
{
    int aggregate = 0;
    int laneID = threadIdx.x % warpSize;
    int start_index_for_warp = blockIdx.x * warpSize * widthFactor;
    int val;
    int add;

    for (unsigned char i = 0; i < widthFactor; i++)
    {
        val = arr[start_index_for_warp + i * warpSize + laneID];
        for (int j = 1; j < warpSize; j <<= 1)
        {
            int predicate = ((int) laneID - j >= 0);
            unsigned mask = __ballot_sync(0xFFFFFFFF, ((int)laneID - j >= 0));
            add = __shfl_up_sync(mask, val, j);
            val += predicate * add;
        }
        if (laneID == warpSize - 1)
        {
            aggregate += val;
        }
    }
    __syncthreads();
    if (threadIdx.x == warpSize - 1)
    {
        ws[blockIdx.x] = aggregate;
    }
}

__global__ void prefix_sum(int *arr, int *n, int *ws)
{
    int laneID = threadIdx.x;
    int start_index_for_warp = blockIdx.x * warpSize * widthFactor;
    int val;
    int add;
    int add_from_previous_computation_of_32_elements = 0;

    int global_level_aggregate = 0;
    if (blockIdx.x > 0)
        global_level_aggregate = ws[blockIdx.x - 1];
    
    for (unsigned char i = 0; i < widthFactor; i++)
    {
        val = arr[start_index_for_warp + i * warpSize + laneID];
        for (int j = 1; j < warpSize; j <<= 1)
        {
            int predicate = ((int) laneID - j >= 0);
            unsigned mask = __ballot_sync(0xFFFFFFFF, ((int)laneID - j >= 0));
            add = __shfl_up_sync(mask, val, j);
            val += predicate * add;
        }
        arr[start_index_for_warp + i * warpSize + laneID] = val + add_from_previous_computation_of_32_elements + global_level_aggregate;
        add_from_previous_computation_of_32_elements += __shfl_sync(0xFFFFFFFF, val, warpSize - 1); // we can change the mask value later, but for now
    }
}

__global__ void add_global_aggregates(int *arr, int *d_n, int *ws)
{
    int laneID = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    // for the start index, we do 1 + blockIdx.x since the first block doesn't need to add an aggregate
    int start_index_for_warp = (1 + blockIdx.x) * warpSize * widthFactor + widthFactor * warpSize * warpID;
    for (unsigned char i = 0; i < widthFactor; i++)
        arr[start_index_for_warp + i * warpSize + laneID] += ws[blockIdx.x];
}

// here n represents distance allowed
__global__ void kogge_stone(int *write_start, int *ws, int *offset_access)
{
    __shared__ int arr[CPUwarpSize * widthFactor];
    int offset = *offset_access;
    int limit = *write_start;
    for (int i = 0; i < widthFactor * warpSize; i += warpSize)
    {
        if (blockIdx.x * warpSize * widthFactor + i + threadIdx.x + offset < limit)
        {
            arr[i + threadIdx.x] = ws[blockIdx.x * warpSize * widthFactor + i + threadIdx.x + offset];
        }
    }
    __syncthreads();
    // kogge-stone computation for separate tiles of size "threads"

    int temp;
    for (int i = 0; i < widthFactor * warpSize; i += warpSize)
    {
        for (int j = 1; j < warpSize; j *= 2)
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
    for (int i = warpSize; i < widthFactor * warpSize; i += warpSize)
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
    int start_index = *start + blockIdx.x * warpSize * widthFactor;
    int lim = *limit;
    int toAdd = (blockIdx.x == 0) ? 0 : *(ws + *limit + blockIdx.x - 1);
    for (int i = 0; i < warpSize * widthFactor; i += warpSize)
    {
        if (start_index + i + threadIdx.x < lim)
        {
            ws[start_index + i + threadIdx.x] += toAdd;
        }
        else
        {
            break; // remove, might improve runtime
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