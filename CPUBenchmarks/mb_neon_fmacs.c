#include "mb_neon_fmacs.h"

// Mixed NEON SIMD and FMACS
void microbenchmark_neon_fmacs(int thread_num) {
	int i;
	for(i = 0; i < NUM_LOOPS; i++) {
		asm("vadd.f32 q6, q5, q4" ::: "q6");
		asm("vadd.f32 q9, q8, q7" ::: "q9");
		asm("vadd.f32 q12, q11, q10" ::: "q12");
		asm("vadd.f32 q15, q14, q13" ::: "q15");
		asm("fmacs s2, s1, s0" ::: "s2");
		asm("fmacs s5, s4, s3" ::: "s5");
		asm("fmacs s8, s7, s6" ::: "s8");
		asm("fmacs s11, s10, s9" ::: "s11");
	}
}

