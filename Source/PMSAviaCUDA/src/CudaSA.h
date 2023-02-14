#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif


#include "SA.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "curand_kernel.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h" // dummy header to get rid of red underline error
#include <assert.h>
//#include <curand_mtgp32_host.h>

// GPU random number generator for each threads, using MTGP32
__constant__ int DF[1000];
__device__ float dev_rnd_0to1(int Idx, curandState* dev_random);
//__device__ float dev_rnd_0to1(int Idx, curandStateMtgp32_t* dev_random);
__device__ int dev_rnd_0toN(int Idx, curandState* dev_random, int N);
//__device__ int dev_rnd_0toN(int Idx, curandStateMtgp32_t* dev_random, int N);
__global__ void xorwow_init_kernel(curandStateXORWOW_t* dev_random, unsigned long long seed, size_t size);
//__device__ void cubmin(int* d_in, int& min_cost, int& min_iter, int num_items);
__device__ void permutation(int* sol, int dim, curandState* dev_random, int Idx);
__device__ int calculate_cost(const int dim, int* D, int* F, int* sol);
__device__ int calculate_delta_cost(const int dim, int* D, int* F, int* sol, int& s1, int& s2);
__device__ void index2swap_2opt(int idx, curandState* dev_random, int dim, int& s1, int& s2);
__device__ void update_solution(int* sol, int s1, int s2);
__device__ void copy_solution(int dim, int* sol1, int* sol2);
__device__ void SA_accept(int idx, curandState* dev_random,
    int* D, int* F, const int dim, int* sol, int* sol_best, int s1, int s2,
    float T_now, int& CurrentCost, int& BestCost);
__global__ void SolveQAP_via_trialset(
    curandState* dev_random, const int dim, int* D, int* F, int* sol, int* BestSolution, int* BestCost,
    const float T_0, const float T_end, const float alpha, const int NumOfRepetation);
__global__ void SolveQAP_final(
    curandState* dev_random, const int dim, int number_of_threads, int* D, int* F, int* BestSolutionArray, int* BestCostArray,
    const float T_0, const float T_end, const float alpha, const int NumOfRepetation);
__global__ void AcquireBestSolution(int* BestSolution, int* BestSolutionArray, int dim, int min_iter);




