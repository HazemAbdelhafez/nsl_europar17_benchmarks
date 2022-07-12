#ifndef MB_STORE_SIMPLE_H
#define MB_STORE_SIMPLE_H

#include "common.h"

typedef struct {
	int threads;
	unsigned int size;
	unsigned int **buffer;
} data_store_simple;

data_store_simple STORE_SIMPLE_DATA;

// Generate the required benchmark data
void prepare_store_simple_data(unsigned int size, int threads);

void prepare_store_simple_thread(int thread_num);

void delete_store_simple_thread(int thread_num);

// Deallocate memory
void delete_store_simple_data();

// Store Benchmark
void microbenchmark_store_simple(int thread_num);

#endif

