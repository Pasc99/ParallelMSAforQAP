#pragma once
#include "CudaSA.h"

__device__ float dev_rnd_0to1(int Idx, curandState* dev_random)
{
    return curand_uniform(&dev_random[Idx]);
}

__device__ int dev_rnd_0toN(int Idx, curandState* dev_random, int N)
{
    return (int)(N * curand_uniform(&dev_random[Idx])) % N;
}

__global__ void xorwow_init_kernel(curandStateXORWOW_t* dev_random, unsigned long long seed, size_t size)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size) return;
    curand_init(seed, tid, 0, &dev_random[tid]);
}

__device__ void permutation(int* sol, int dim, curandState* dev_random, int Idx)
{
    for (int i = 0; i < dim; i++) {
        sol[i] = i;
    }
    for (int i = dim - 1; i >= 0; i--) {
        int rnd = dev_rnd_0toN(Idx, dev_random, dim);
        int tmp = sol[i];
        sol[i] = sol[rnd];
        sol[rnd] = tmp;
    }
}

__device__ int calculate_cost(const int dim, int* D, int* F, int* sol)
{
    int cost = 0;
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
            cost += D[i * dim + j] * F[sol[i] * dim + sol[j]];
    return cost;
}

__device__ int calculate_delta_cost(const int dim, int* D, int* F, int* sol, int& s1, int& s2)
{
    //D[i * dim + j]
    //F[i * dim + j]
    //int delta_cost = (F[sol[s1] * dim + sol[s1]] - F[sol[s2] * dim + sol[s1]]) * (D[s2 * dim + s1] - D[s1 * dim + s2]);
    int delta_cost = 0;
    for (size_t i = 0; i < dim; i++)
        if (i != s1 && i != s2)
            delta_cost += (F[sol[i] * dim + sol[s1]] - F[sol[i] * dim + sol[s2]])
            * (D[i * dim + s2] - D[i * dim + s1])
            + (F[sol[s1] * dim + sol[i]] - F[sol[s2] * dim + sol[i]])
            * (D[s2 * dim + i] - D[s1 * dim + i]);
    return delta_cost;
}

__device__ void index2swap_2opt(int idx, curandState* dev_random, int dim, int& s1, int& s2)
{
    s1 = dev_rnd_0toN(idx, dev_random, dim);
    do
    {
        s2 = dev_rnd_0toN(idx, dev_random, dim);
    } while (s1 == s2);
}

__device__ void update_solution(int* sol, int s1, int s2)
{
    int tmp = sol[s1];
    sol[s1] = sol[s2];
    sol[s2] = tmp;
}

__device__ void copy_solution(int dim, int* sol1, int* sol2)
{
    for (size_t i = 0; i < dim; i++)
    {
        sol2[i] = sol1[i];
    }
}

__device__ void SA_accept(int idx, curandState* dev_random,
    int* D, int* F, const int dim, int* sol, int* sol_best, int s1, int s2,
    float T_now, int& CurrentCost, int& BestCost)
{
    int delta_cost = calculate_delta_cost(dim, D, F, sol, s1, s2);
    float probablity_to_accept = exp(-(float)delta_cost / T_now);

    if ((delta_cost < 0) || (dev_rnd_0to1(idx, dev_random) <= probablity_to_accept))
    {
        update_solution(sol, s1, s2);
        CurrentCost += delta_cost;
    }
    if (BestCost > CurrentCost)
    {
        copy_solution(dim, sol, sol_best);
        BestCost = CurrentCost;
    }

};

// PMSA/CUDA with pre-generated trial set
// Not used in the final paper, examinate it before using it plz
__global__ void SolveQAP_via_trialset(
    curandState* dev_random, const int dim, int* D, int* F, int* sol, int* BestSolution, int* BestCost,
    const float T_0, const float T_end, const float alpha, const int NumOfRepetation)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* localSol = new int[dim];
    int* localBestSol = new int[dim];
    for (size_t i = 0; i < dim; i++)
        localSol[i] = sol[tid * dim + i];
    __syncthreads();

    /*printf("tid = %d, sol = ", tid);
    for (size_t i = 0; i < dim; i++)
    {
        printf("%d ", localSol[i]);
    }
    printf("\n");*/

    int CurrentCost = calculate_cost(dim, D, F, localSol);
    int LocalBestCost = INT_MAX;

    int s1 = 0, s2 = 0;
    for (float T_now = T_0; T_now >= T_end; T_now *= alpha)
        for (int re = 0; re < NumOfRepetation; re++)
        {
            index2swap_2opt(tid, dev_random, dim, s1, s2);
            SA_accept(tid, dev_random, D, F, dim, localSol, localBestSol, s1, s2, T_now, CurrentCost, LocalBestCost);
        }
    __syncthreads();

    //printf("\ntid = %d, CurrentCost = %d\n", tid, CurrentCost);

    for (size_t i = 0; i < dim; i++)
        BestSolution[tid * dim + i] = localBestSol[i];
    BestCost[tid] = LocalBestCost;

    delete[] localSol;
    delete[] localBestSol;
}

__global__ void AcquireBestSolution(int* BestSolution, int* BestSolutionArray, int dim, int min_iter)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == min_iter)
    {
        for (size_t i = 0; i < dim; i++)
            BestSolution[i] = BestSolutionArray[tid * dim + i];
    }
}

// Generates initial solutions for each thread
__global__ void SolveQAP_final(
    curandState* dev_random, const int dim, int number_of_threads, int* D, int* F, int* BestSolutionArray, int* BestCostArray,
    const float T_0, const float T_end, const float alpha, const int NumOfRepetation)
{
    int uid = threadIdx.x + blockIdx.x * blockDim.x;

    int* localSol = new int[dim];
    int* localBestSol = new int[dim];
    permutation(localSol, dim, dev_random, uid);
    __syncthreads();

    /*for (size_t j = 0; j < number_of_threads; j++)
    {
        if (uid == j)
        {
            printf("uid = %d, sol = ", uid);
            for (size_t i = 0; i < dim; i++)
            {
                printf("%d ", localSol[i]);
            }
            printf("\n");
        }
        __syncthreads();
    }*/

    int CurrentCost = calculate_cost(dim, D, F, localSol);
    int LocalBestCost = INT_MAX;

    int s1 = 0, s2 = 0;
    for (float T_now = T_0; T_now >= T_end; T_now *= alpha)
        for (int re = 0; re < NumOfRepetation; re++)
        {
            index2swap_2opt(uid, dev_random, dim, s1, s2);
            SA_accept(uid, dev_random, D, F, dim, localSol, localBestSol, s1, s2, T_now, CurrentCost, LocalBestCost);
        }
    __syncthreads();

    BestCostArray[uid] = LocalBestCost;
    for (size_t i = 0; i < dim; i++)
        BestSolutionArray[uid * dim + i] = localBestSol[i];
    __syncthreads();

    delete[] localSol;
    delete[] localBestSol;
};