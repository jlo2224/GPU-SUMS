#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define threads 256
#define logthreads 8
#define widthFactor 4
#define FULL_MASK 0xffffffff
// make threads a power of 2

void inclusive_scan(int *arr, int n, int *result);
__global__ void prefix_sum(int *arr, int *d_n, int *d_ws);
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
        arr[i] = rand() % 5;
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
    int *d_ws;
    int ws_size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    ////////////////// TIMER START
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int *d_n;

    int blocks = (n + threads * widthFactor - 1) / (threads * widthFactor);
    ws_size = blocks + 2 * ((blocks + threads * widthFactor - 1) / (threads * widthFactor));
    int f = 0;
    cudaMalloc((void **)&d_ws, ws_size * sizeof(int));
    cudaMemcpy(d_ws, &f, sizeof(int), cudaMemcpyHostToDevice);
    d_ws++;
    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    prefix_sum<<<blocks, threads>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();

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
        j = (i + threads * widthFactor - 1) / (threads * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        kogge_stone<<<j, threads>>>(limit, d_ws, offset_access);
        cudaDeviceSynchronize();
        arr_k[to_write] = k;
        arr_f[to_write] = f;
        to_write++;
        f = k;
        k += j;
    }
    // unroll the tree, and don't forget to do it in reverse order
    for (int i = to_write - 1; i >= 0; i--)
    {
        k = arr_k[i];
        f = arr_f[i];
        j = (k - f + threads * widthFactor - 1) / (threads * widthFactor);
        cudaMemcpy(limit, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offset_access, &f, sizeof(int), cudaMemcpyHostToDevice);
        add_constant<<<j, threads>>>(limit, d_ws, offset_access);
        cudaDeviceSynchronize();
    }
    // verify_ws_update<<<1, 1>>>(d_arr, d_ws, d_n);
    // cudaDeviceSynchronize();
    // print_line();

    downsweep<<<blocks, threads>>>(d_arr, d_n, d_ws);
    cudaDeviceSynchronize();

    /*
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
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

__global__ void prefix_sum(int *arr, int *n, int *ws)
{
    __shared__ int mini[threads / 32];
    int parent_index = blockIdx.x * threads * widthFactor + threadIdx.x * widthFactor;
    int local_sum = 0;
    for (unsigned char i = 0; i < widthFactor; i++)
    {
        local_sum += arr[parent_index + i];
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }
    if (threadIdx.x % 32 == 0)
        mini[threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 8) // should be half of the size of mini, which in this case is 8. Shouldn't hardcode it though.
    {
        local_sum = mini[threadIdx.x];
        for (int offset = 4; offset > 0; offset /= 2)
        {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
    }
    if (threadIdx.x == 0)
        ws[blockIdx.x] = local_sum;
    /*
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < threads / 32; i++)
        {
            sum += mini[i];
        }
        ws[blockIdx.x] = sum;
    }
    */
}

__global__ void downsweep(int *arr, int *d_n, int *ws) // some access of shared memory has bank conflicts
{
    __shared__ int heap[2 * threads];
    int left, right, and_mask, parent_index;
    int position = (threadIdx.x + 1) % threads; // we do this so we need not subtract by 1, thus avoiding bank conflicts, particularly for final prefix sum of widthFactor size serially in registers, but declared here for other use within the function
    parent_index = blockIdx.x * threads * widthFactor + threadIdx.x * widthFactor;
    heap[threadIdx.x] = 0; // try changing to see if bank conflicts
    heap[threadIdx.x + threads] = arr[parent_index];
    for (unsigned char i = 1; i < widthFactor; i++)
    {
        heap[threadIdx.x + threads] += arr[parent_index + i];
    }
    __syncthreads(); // we can probably remove this syncthreads call since lock-step.
    // Only needs to be done if threadIdx.x % 2 == 0, but might as well compute it for the rest since there is no additional runtime
    left = heap[threadIdx.x + threads];
    right = heap[position + threads];
    and_mask = 1;
    parent_index = (threads + threadIdx.x) >> 1;

    // printf("%d\n", __ballot_sync(FULL_MASK, threadIdx.x % 4 == 0));
    for (unsigned char i = 0; i < 4; i++)
    {
        /*
        if ((threadIdx.x % pow2 == 0) != (threadIdx.x & and_mask == 0))
            printf("%d, %d, %d, %d\n", pow2, and_mask, (threadIdx.x % pow2 == 0), (threadIdx.x & and_mask == 0));
            */
        unsigned mask = __ballot_sync(FULL_MASK, !(threadIdx.x & and_mask));
        if (!(threadIdx.x & and_mask))
        {
            left += right;
            heap[parent_index] = left;
        }
        __syncwarp(mask);
        right = heap[parent_index + 1];
        parent_index = parent_index >> 1;
        and_mask <<= 1;
        and_mask += 1;
    }

    __syncthreads();

    for (unsigned char i = 4; i < logthreads; i++)
    {
        if (!(threadIdx.x & and_mask))
        {
            left += right;
            heap[parent_index] = left;
        }
        __syncthreads();
        right = heap[parent_index + 1];   // reassign for next layer
        parent_index = parent_index >> 1; // reassign for next layer
        and_mask <<= 1;
        and_mask += 1;
    }

    int start = 1;
    for (unsigned char i = 0; i < logthreads; i++)
    {
        if (threadIdx.x % (threads / start) == 0)
        {
            int parent = heap[start + threadIdx.x * start / threads];
            heap[(start + threadIdx.x * start / threads) * 2 + 1] = parent;
            if (threadIdx.x * start / threads + 1 != start)
            {
                heap[(start + threadIdx.x * start / threads) * 2 + 2] += parent; // no need sync here for downsweep; threads do not interact with each other
            }
        }
        start *= 2;
        __syncthreads();
    }

    // first of 4 (widthFactor) to be updated. CHANGE THIS SO HEAP IS ADDED TO INSTEAD, THUS REDUCING GLOBAL MEMORY ACCESSES
    int writeTo = blockIdx.x * threads * widthFactor + position * widthFactor;
    arr[writeTo] += *(ws + blockIdx.x - 1);
    if (position != 0)
    {
        arr[writeTo] += heap[threads + threadIdx.x]; // optimized this by doing +1 to global array rather than -1 for shared memory, thus avoiding bank conflicts
    }
    for (unsigned char i = 1; i < widthFactor; i++)
    {
        arr[writeTo + i] += arr[writeTo + i - 1];
    }
}

// here n represents distance allowed
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
    int total = source[0];
    int returnValue = 0;
    for (int i = 1; i < n; i++)
    {
        total += *(source + i);
        if (total != *(answer + i))
        {
            returnValue = i;
            break;
        }
    }
    if (returnValue != 0)
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