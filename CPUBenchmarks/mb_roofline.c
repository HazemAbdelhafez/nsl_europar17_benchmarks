#include "mb_roofline.h"

// Allocate required memory
void prepare_roofline_data(int fp_mem_ratio, int threads) {
	ROOFLINE_DATA.threads = threads;
	ROOFLINE_DATA.fp_mem_ratio = fp_mem_ratio;
	ROOFLINE_DATA.fp_loops = NUM_LOOPS;
	ROOFLINE_DATA.store_size = ROOFLINE_STORE_SIZE;

	prepare_neon_data(ROOFLINE_DATA.fp_loops);
	prepare_store_long_unroll_data(ROOFLINE_DATA.store_size, threads);
}

void prepare_roofline_data_size(int fp_mem_ratio, int threads,
		int fp_loops, int store_size) {
	ROOFLINE_DATA.threads = threads;
	ROOFLINE_DATA.fp_mem_ratio = fp_mem_ratio;
	ROOFLINE_DATA.fp_loops = fp_loops;
	ROOFLINE_DATA.store_size = store_size;

	prepare_neon_data(ROOFLINE_DATA.fp_loops);
	prepare_store_long_unroll_data(ROOFLINE_DATA.store_size, threads);
}

void prepare_roofline_thread(int thread_num) {
	prepare_store_long_unroll_thread(thread_num);
}

void delete_roofline_thread(int thread_num) {
	delete_store_long_unroll_thread(thread_num);
}

// Deallocate memory
void delete_roofline_data() {
	delete_store_long_unroll_data();
}

// Roofline model benchmark
void microbenchmark_roofline(int thread_num) {
	int i;
	for (i = ROOFLINE_LOOPS * ROOFLINE_DATA.fp_mem_ratio; i > 0; i--)
		microbenchmark_neon(thread_num);

	for (i = ROOFLINE_LOOPS; i > 0; i--)
		microbenchmark_store_long_unroll(thread_num);
}

