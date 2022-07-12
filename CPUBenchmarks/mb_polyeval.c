#include "mb_polyeval.h"

// Generate required polynomials
void prepare_poly_eval_data(int dim, int threads) {
	POLY_EVAL_DATA.threads = threads;
	POLY_EVAL_DATA.len = dim;

	POLY_EVAL_DATA.x = safe_malloc(sizeof(float)*threads);
	POLY_EVAL_DATA.c = safe_malloc(sizeof(float*)*threads);
}

void prepare_poly_eval_thread(int thread_num) {
	POLY_EVAL_DATA.x[thread_num] = (float)rand()/(float)RAND_MAX;
	POLY_EVAL_DATA.c[thread_num] =
		safe_malloc(sizeof(float)*POLY_EVAL_DATA.len);
	int i;
	for (i = 0; i < POLY_EVAL_DATA.len; i++) {
		POLY_EVAL_DATA.c[thread_num][i] = (float)rand()/(float)RAND_MAX;
	}
}

void delete_poly_eval_thread(int thread_num) {
	free(POLY_EVAL_DATA.c[thread_num]);
}

// Deallocate memory assigned to coefficients
void delete_poly_eval_data() {
	free(POLY_EVAL_DATA.x);
	free(POLY_EVAL_DATA.c);
}

void microbenchmark_poly_eval(int thread_num) {
	int i;
	volatile float res = POLY_EVAL_DATA.c[thread_num][POLY_EVAL_DATA.len - 1];

	for (i = POLY_EVAL_DATA.len - 1; i > 0; i--) {
		res = POLY_EVAL_DATA.c[thread_num][i - 1] +
			POLY_EVAL_DATA.x[thread_num] * res;
	}
}

