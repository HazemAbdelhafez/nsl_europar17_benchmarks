#include "utils.h"


float convert_to_sec(long long time_usec) {
	return ((float)time_usec * 1.0e-06);
}
float calculate_average(float* array, int repeat) {
	int i;
	float sum = 0;

	for (i = 1; i < repeat; i++) {
		sum += array[i];
	}

	float average = sum / (double)(repeat-1);

	return average;
}
 // Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
 void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void fill_2d_matrix(float* matrix, int n, int m, float value) {
	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			matrix[i*m+j] = value;
}

void print_matrix(const float *A, int rows, int cols) {
 for(int i = 0; i < rows; i++){
	 for(int j = 0; j < cols; j++){
		 printf("%f ", A[i*cols + j]);
	 }
	 printf("\n");
 }
 printf("\n");
}
