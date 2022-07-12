#ifndef MB_TILED_MM_H
#define MB_TILED_MM_H

#include "common.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef struct {
	int threads;
	int dim;
	int block_size;
	float **a, **b, **c;
} data_tiled_mm;

data_tiled_mm TILED_MM_DATA;

// Generate required matrices and scaling factors
void prepare_tiled_mm_data(int dim, int block_size, int threads);

void prepare_tiled_mm_thread(int thread_num);

void delete_tiled_mm_thread(int thread_num);

// Deallocate memory assigned to matrices
void delete_tiled_mm_data();

// Matrix Multiply (SGEMM)
void microbenchmark_tiled_mm(int thread_num);

#endif

