#include "mb_fmacs.h"

// Fused multiply-add (FMACS)
void microbenchmark_fmacs(int thread_num) {
	int i;
	for(i = 0; i < NUM_LOOPS; i++) {
		asm("fmacs s14, s13, s12" ::: "s14");
		asm("fmacs s17, s16, s15" ::: "s17");
		asm("fmacs s20, s19, s18" ::: "s20");
		asm("fmacs s23, s22, s21" ::: "s23");
	}
}

