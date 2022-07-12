#ifndef MB_SGEMM_H
#define MB_SGEMM_H

#include <cblas.h>

#include "common.h"

typedef struct {
	int threads;
	int dim;
	float *alpha, *beta;
	float **a, **b, **c;
} data_sgemm;

data_sgemm SGEMM_DATA;

// Generate required matrices and scaling factors
void prepare_sgemm_data(int dim, int threads);

void prepare_sgemm_thread(int thread_num);

void delete_sgemm_thread(int thread_num);

// Deallocate memory assigned to matrices
void delete_sgemm_data();

// Matrix Multiply (SGEMM)
void microbenchmark_sgemm(int thread_num);

#endif

