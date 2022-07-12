#include "mb_store_simple.h"

// Generate the required benchmark data
void prepare_store_simple_data(unsigned int size, int threads) {
	STORE_SIMPLE_DATA.threads = threads;
	STORE_SIMPLE_DATA.size = size;
	STORE_SIMPLE_DATA.buffer = safe_malloc(sizeof(int*)*threads);
}

void prepare_store_simple_thread(int thread_num) {
	STORE_SIMPLE_DATA.buffer[thread_num] =
		safe_malloc(sizeof(int)*STORE_SIMPLE_DATA.size);
}

void delete_store_simple_thread(int thread_num) {
	free(STORE_SIMPLE_DATA.buffer[thread_num]);
}

// Deallocate memory
void delete_store_simple_data() {
	free(STORE_SIMPLE_DATA.buffer);
}

// Store Benchmark
void microbenchmark_store_simple(int thread_num) {
	asm volatile(
		"mov r0, %0\n\t"
		"mov r1, %1\n"
		"again:\n\t"
		"str r1, [r0], #4\n\t"
		"subs r1, r1, #1\n\t"
		"bne again"
		:: "r" (STORE_SIMPLE_DATA.buffer[thread_num]),
		"r" (STORE_SIMPLE_DATA.size)
		: "r0", "r1", "memory"
	);
}

