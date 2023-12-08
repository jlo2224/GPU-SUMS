#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cub/cub.cuh>

__global__ void prefix_sum(int *arr, int *d_n);
__global__ void add_constant(int *arr, int *toAdd, int *d_n);
void printArr(int *arr, int n);
void checkArr(int *source, int *answer, int n);

int main(int args, char *argv[])
{
    int *arr;
    int *result;
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
    arr = (int *)malloc(n * sizeof(int));
    result = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 5;
    }

    

    int *d_arr;
    cudaMalloc((void**) &d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n*sizeof(int), cudaMemcpyHostToDevice);
    int *d_result;
    cudaMalloc((void**) &d_result, n * sizeof(int));
    cudaMemcpy(d_result, result, n*sizeof(int), cudaMemcpyHostToDevice);

    size_t scan_workspace_size = 0;
    void *scan_workspace = NULL;
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    cub::DeviceScan::InclusiveSum(scan_workspace, scan_workspace_size, d_arr, d_result, n);
    cudaMalloc(&scan_workspace, scan_workspace_size);
    cub::DeviceScan::InclusiveSum(scan_workspace, scan_workspace_size, d_arr, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_arr);

    
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("(%d, %f)\n", n, milliseconds); // Change to measure GB/s rather than ms
    //printArr(arr, n);
    //checkArr(arr, result, n);

    free(arr);
    free(result);
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
    int total = *(source);
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
