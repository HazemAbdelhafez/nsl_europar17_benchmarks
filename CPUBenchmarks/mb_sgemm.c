#include "mb_sgemm.h"

// Generate required matrices and scaling factors
void prepare_sgemm_data(int dim, int threads) {
	SGEMM_DATA.threads = threads;
	SGEMM_DATA.dim = dim;

	SGEMM_DATA.alpha = safe_malloc(sizeof(float)*threads);
	SGEMM_DATA.beta = safe_malloc(sizeof(float)*threads);

	SGEMM_DATA.a = safe_malloc(sizeof(float*)*threads);
	SGEMM_DATA.b = safe_malloc(sizeof(float*)*threads);
	SGEMM_DATA.c = safe_malloc(sizeof(float*)*threads);
}

void prepare_sgemm_thread(int thread_num) {
	// Generate input matrices
	random_2d_matrix(&SGEMM_DATA.a[thread_num], SGEMM_DATA.dim, SGEMM_DATA.dim);
	random_2d_matrix(&SGEMM_DATA.b[thread_num], SGEMM_DATA.dim, SGEMM_DATA.dim);
	random_2d_matrix(&SGEMM_DATA.c[thread_num], SGEMM_DATA.dim, SGEMM_DATA.dim);

	// Generate scaling factors
	SGEMM_DATA.alpha[thread_num] = (float)rand()/(float)RAND_MAX;
	SGEMM_DATA.beta[thread_num] = (float)rand()/(float)RAND_MAX;
}

void delete_sgemm_thread(int thread_num) {
	free(SGEMM_DATA.a[thread_num]);
	free(SGEMM_DATA.b[thread_num]);
	free(SGEMM_DATA.c[thread_num]);
}

// Deallocate memory assigned to matrices
void delete_sgemm_data() {
	free(SGEMM_DATA.alpha);
	free(SGEMM_DATA.beta);

	free(SGEMM_DATA.a);
	free(SGEMM_DATA.b);
	free(SGEMM_DATA.c);
}

// Matrix Multiply (SGEMM)
void microbenchmark_sgemm(int thread_num) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, SGEMM_DATA.dim,
			SGEMM_DATA.dim, SGEMM_DATA.dim, SGEMM_DATA.alpha[thread_num],
			SGEMM_DATA.a[thread_num], SGEMM_DATA.dim, SGEMM_DATA.b[thread_num],
			SGEMM_DATA.dim, SGEMM_DATA.beta[thread_num], SGEMM_DATA.c[thread_num],
			SGEMM_DATA.dim);
}

