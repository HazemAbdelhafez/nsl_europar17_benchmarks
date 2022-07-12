#ifndef MB_STORE_UNROLL_H
#define MB_STORE_UNROLL_H

#include "common.h"

typedef struct {
	int threads;
	unsigned int size;
	unsigned int **buffer;
} data_store_unroll;

data_store_unroll STORE_UNROLL_DATA;

// Generate the required benchmark data
void prepare_store_unroll_data(unsigned int size, int threads);

void prepare_store_unroll_thread(int thread_num);

void delete_store_unroll_thread(int thread_num);

// Deallocate memory
void delete_store_unroll_data();

// Store Benchmark
void microbenchmark_store_unroll(int thread_num);

#endif

