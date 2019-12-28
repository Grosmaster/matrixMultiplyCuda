#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE  32
#define N           3200        

__global__ void matMult(float* a, float* b, int n, float* c)
{

    int   bx = blockIdx.x;    
    int   by = blockIdx.y;
    int   tx = threadIdx.x;        
    int   ty = threadIdx.y;
    float sum = 0.0f;           
    int   ia = n * BLOCK_SIZE * by + n * ty;   
    int   ib = BLOCK_SIZE * bx + tx;

    
    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];

    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    c[ic + n * ty + tx] = sum;
}


int main(int argc, char* argv[])
{
    printf("START\n");
    int numBytes = N * N * sizeof(float);

    float* a = new float[N * N];
    float* b = new float[N * N];
    float* c = new float[N * N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int	k = N * i + j;
            a[k] = 1.0f;
            b[k] = 1.0f;
        }

    float* adev = NULL;
    float* bdev = NULL;
    float* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);


    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMult << <blocks, threads >> > (adev, bdev, N, cdev);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    printf("END %d", c[1]);

    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);

    delete a;
    delete b;
    delete c;

    return 0;
}
