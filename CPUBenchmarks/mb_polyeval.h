#ifndef MB_POLYEVAL_H
#define MB_POLYEVAL_H

#include "common.h"

typedef struct {
	int threads;
	int len;
	float *x;
	float **c;
} data_poly_eval;

data_poly_eval POLY_EVAL_DATA;

void prepare_poly_eval_data(int dim, int threads);

void prepare_poly_eval_thread(int thread_num);

void delete_poly_eval_thread(int thread_num);

// Deallocate memory assigned to coefficients
void delete_poly_eval_data();

void microbenchmark_poly_eval(int thread_num);

#endif

