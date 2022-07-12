#include "mb_neon.h"

// Setup benchmark parameters
void prepare_neon_data(int loops) {
	NEON_DATA.loops = loops;
}

// NEON SIMD (floating point)
void microbenchmark_neon(int thread_num) {
	int i;
	for(i = NEON_DATA.loops; i > 0; i--) {
		asm("vadd.f32 q2, q1, q0" ::: "q2");
		asm("vadd.f32 q5, q4, q3" ::: "q5");
		asm("vadd.f32 q8, q7, q6" ::: "q8");
		asm("vadd.f32 q11, q10, q9" ::: "q11");
	}
}

