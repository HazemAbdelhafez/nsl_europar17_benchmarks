#include "mb_tiled_mm.h"

// Generate required matrices and scaling factors
void prepare_tiled_mm_data(int dim, int block_size, int threads) {
	TILED_MM_DATA.threads = threads;
	TILED_MM_DATA.dim = dim;
	TILED_MM_DATA.block_size = block_size;

	TILED_MM_DATA.a = safe_malloc(sizeof(float*)*threads);
	TILED_MM_DATA.b = safe_malloc(sizeof(float*)*threads);
	TILED_MM_DATA.c = safe_malloc(sizeof(float*)*threads);
}

void prepare_tiled_mm_thread(int thread_num) {
	// Generate input matrices
	fill_2d_matrix(&TILED_MM_DATA.a[thread_num], TILED_MM_DATA.dim,
			TILED_MM_DATA.dim, 1.0f);
	fill_2d_matrix(&TILED_MM_DATA.b[thread_num], TILED_MM_DATA.dim,
			TILED_MM_DATA.dim, 2.0f);
	fill_2d_matrix(&TILED_MM_DATA.c[thread_num], TILED_MM_DATA.dim,
			TILED_MM_DATA.dim, 0.0f);
}

void delete_tiled_mm_thread(int thread_num) {
	free(TILED_MM_DATA.a[thread_num]);
	free(TILED_MM_DATA.b[thread_num]);
	free(TILED_MM_DATA.c[thread_num]);
}

// Deallocate memory assigned to matrices
void delete_tiled_mm_data() {
	free(TILED_MM_DATA.a);
	free(TILED_MM_DATA.b);
	free(TILED_MM_DATA.c);
}

// Matrix Multiply (tiled_mm)
void microbenchmark_tiled_mm(int thread_num) {
	float *a = TILED_MM_DATA.a[thread_num];
	float *b = TILED_MM_DATA.b[thread_num];
	float *c = TILED_MM_DATA.c[thread_num];
	int block_size = TILED_MM_DATA.block_size;
	int dim = TILED_MM_DATA.dim;

	int i_outer;
	for (i_outer = 0; i_outer < dim; i_outer += block_size) {
		int j_outer;
		for (j_outer = 0; j_outer < dim; j_outer += block_size) {
			int k_outer;
			for (k_outer = 0; k_outer < dim; k_outer += block_size) {
				int i_inner;
				int i_max = i_outer + block_size;
				for(i_inner = i_outer; i_inner < MIN(i_max, dim); i_inner++) {
					int j_inner;
					int j_max = j_outer + block_size;
					for(j_inner = j_outer; j_inner < MIN(j_max, dim); j_inner++) {
						int k_inner;
						int k_max = k_outer + block_size;
						float sum = 0.0f;
						for(k_inner = k_outer; k_inner < MIN(k_max, dim); k_inner++) {
							sum += a[i_inner*dim + k_inner] *
								b[k_inner*dim + j_inner];
						}
							c[i_inner*dim + j_inner] = sum;
					}
				}
			}
		}
	}

	print_2d_matrix(a, dim, dim);
	print_2d_matrix(b, dim, dim);
	print_2d_matrix(c, dim, dim);
}

