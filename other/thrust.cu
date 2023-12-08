#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <thrust/scan.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

__global__ void prefix_sum(int *arr, int *d_n);
__global__ void add_constant(int *arr, int *toAdd, int *d_n);
void printArr(int *arr, int n);
void checkArr(int *source, int *answer, int n);

int main(int args, char *argv[])
{
    thrust::device_ptr<int> arr;
    thrust::device_ptr<int> result;
    int n;
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (args == 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        n = 1024;
    }
    arr = thrust::device_malloc<int>(n * sizeof(int));
    result = thrust::device_malloc<int>(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 5;
    }

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    thrust::inclusive_scan(arr, arr + n, result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("(%d, %f)\n", n, milliseconds);
    // printArr(arr, n);
    // checkArr(arr, result, n);

    thrust::device_free(arr);
    thrust::device_free(result);
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

void checkArr(int *source, int *answer, int n)
{
    int total = source[0];
    printf("\033[1;32m[");
    for (int i = 1; i < n; i++)
    {
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
