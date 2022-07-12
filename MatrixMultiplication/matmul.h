/*
 * matmul.h
 *
 *  Created on: Dec 14, 2016
 *      Author: hazem
 */

#include "utils.h"

#ifndef MATMUL_H_
#define MATMUL_H_

pthread_t tid[2];
float *matA, *matB, *matC;
float *cpuMatA, *cpuMatB, *cpuMatC;
float *gpuMatA, *gpuMatB, *gpuMatC;
float *cmatA, *cmatB, *cmatC;
float *mappedMatA, *mappedMatB, *mappedMatC;

int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
int cpuCols, gpuCols;
int dim, repeats, threads;
int showDetails;
cudaStream_t stream1;

// Matrix multiplication GPU kernels
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void gpu_blas_split(const float *A, const float *B, float *C, const int m, const int k, const int n);
void gpu_blas_split_with_stream(const float *A, const float *B, float *C, const int m, const int k, const int n);

// Matrix multiplication CPU kernels
void cpu_blas_mmul(float *A, float *B, float *C, const int m, const int k, const int n);
void cpu_blas_split(float *A, float *B, float *C, const int m, const int k, const int n);

// GPU Benchmarks
void gpuMatMul();
void gpuMatMulWithUM();
void gpuMatMulWithMemCpy();

// CPU Benchmarks
void cpuMatMul();
void cpuMatMulWithUM();
void cpuMatMultWithHostAlloc();

// Heterogeneous Benchmarks
void heterogeneousWithHostAlloc();
void heterogeneousWithZeroMemCpy();
void heterogeneousWithAsyncMemCpy();
void heterogeneousWithSyncMemCpy();
void heterogeneousWithSplitMem();

// Thread launching helpers
void* gpuThread();
void* gpuThreadWithMemSplit();
void* gpuThreadAsyncMemCpy();
void* gpuThreadSyncMemCpy();
void* cpuThread();
void* cpuThreadWithMemSplit();

#endif /* MATMUL_H_ */
