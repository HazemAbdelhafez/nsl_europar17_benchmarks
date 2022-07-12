#ifndef MB_ROOFLINE_H
#define MB_ROOFLINE_H

#include "common.h"
#include "mb_neon.h"
#include "mb_store_long_unroll.h"

#define ROOFLINE_LOOPS 2
#define ROOFLINE_STORE_SIZE 4000000

typedef struct {
	int threads;
	int fp_mem_ratio;
	int store_size;
	int fp_loops;
} data_roofline;

data_roofline ROOFLINE_DATA;

// Allocate required memory
void prepare_roofline_data(int fp_mem_ratio, int threads);

void prepare_roofline_data_size(int fp_mem_ratio, int threads,
		int fp_loops, int store_size);

void prepare_roofline_thread(int thread_num);

void delete_roofline_thread(int thread_num);

// Deallocate memory
void delete_roofline_data();

// Roofline model benchmark
void microbenchmark_roofline(int thread_num);

#endif

