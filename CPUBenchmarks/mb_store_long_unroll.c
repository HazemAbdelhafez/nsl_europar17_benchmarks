#include "mb_store_long_unroll.h"

// Generate the required benchmark data
void prepare_store_long_unroll_data(unsigned int size, int threads) {
	STORE_LONG_UNROLL_DATA.threads = threads;
	STORE_LONG_UNROLL_DATA.size = size;
	STORE_LONG_UNROLL_DATA.buffer = safe_malloc(sizeof(int*)*threads);
}

void prepare_store_long_unroll_thread(int thread_num) {
	STORE_LONG_UNROLL_DATA.buffer[thread_num] =
		safe_malloc(sizeof(int)*STORE_LONG_UNROLL_DATA.size);
}

void delete_store_long_unroll_thread(int thread_num) {
	free(STORE_LONG_UNROLL_DATA.buffer[thread_num]);
}

// Deallocate memory
void delete_store_long_unroll_data() {
	free(STORE_LONG_UNROLL_DATA.buffer);
}

// Store Benchmark
void microbenchmark_store_long_unroll(int thread_num) {
	asm volatile(
		"mov r0, %0\n\t"
		"mov r1, %1\n"
		"mov r2, %1\n"
		"again:\n\t"
		"strd r1, [r0, #0*8]\n\t"
		"strd r1, [r0, #1*8]\n\t"
		"strd r1, [r0, #2*8]\n\t"
		"strd r1, [r0, #3*8]\n\t"
		"adds r0, r0, #8*4\n\t"
		"subs r1, r1, #2*4\n\t"
		"bne again"
		:: "r" (STORE_LONG_UNROLL_DATA.buffer[thread_num]),
		"r" (STORE_LONG_UNROLL_DATA.size)
		: "r0", "r1", "r2", "memory"
	);
}

