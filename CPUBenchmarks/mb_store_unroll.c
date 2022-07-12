#include "mb_store_unroll.h"

// Generate the required benchmark data
void prepare_store_unroll_data(unsigned int size, int threads) {
	STORE_UNROLL_DATA.threads = threads;
	STORE_UNROLL_DATA.size = size;
	STORE_UNROLL_DATA.buffer = safe_malloc(sizeof(int*)*threads);
}

void prepare_store_unroll_thread(int thread_num) {
	STORE_UNROLL_DATA.buffer[thread_num] =
		safe_malloc(sizeof(int)*STORE_UNROLL_DATA.size);
}

void delete_store_unroll_thread(int thread_num) {
	free(STORE_UNROLL_DATA.buffer[thread_num]);
}

// Deallocate memory
void delete_store_unroll_data() {
	free(STORE_UNROLL_DATA.buffer);
}

// Store Benchmark
void microbenchmark_store_unroll(int thread_num) {
	asm volatile(
		"mov r0, %0\n\t"
		"mov r1, %1\n"
		"again:\n\t"
		"str r1, [r0, #0*4]\n\t"
		"str r1, [r0, #1*4]\n\t"
		"str r1, [r0, #2*4]\n\t"
		"str r1, [r0, #3*4]\n\t"
		"adds r0, r0, #4*4\n\t"
		"subs r1, r1, #1*4\n\t"
		"bne again"
		:: "r" (STORE_UNROLL_DATA.buffer[thread_num]),
		"r" (STORE_UNROLL_DATA.size)
		: "r0", "r1", "memory"
	);
}

