#include "mb_fadds.h"

// Floating point add (FADDS)
void microbenchmark_fadds(int thread_num) {
	int i;
	for(i = 0; i < NUM_LOOPS; i++) {
		asm("fadds s2, s1, s0" ::: "s2");
		asm("fadds s5, s4, s3" ::: "s5");
		asm("fadds s11, s10, s9" ::: "s11");
		asm("fadds s14, s13, s12" ::: "s14");
		asm("fadds s20, s19, s18" ::: "s20");
		asm("fadds s23, s22, s21" ::: "s23");
	}
}

