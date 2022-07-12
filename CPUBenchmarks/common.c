#include "common.h"

// Allocate array using malloc and check if successful
void *safe_malloc(size_t size) {
	void *ptr = malloc(size);

	if (ptr == NULL) {
		fprintf(stderr, "Error: not enough memory for allocation!\n");
		exit(-1);
	}

	return ptr;
}

// Measure the current time in seconds
double get_time_seconds() {
	struct timeval tp;
	struct timezone tzp;

	gettimeofday(&tp, &tzp);

	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Fill 2D Matrix of size NxM with a provided value
void fill_2d_matrix(float** matrix, int n, int m, float value) {
	float* a = safe_malloc(sizeof(float)*n*m);

	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			a[i*m+j] = value;

	*matrix = a;
}

// Create a random 2D Matrix of size NxM
void random_2d_matrix(float** matrix, int n, int m) {
	float* a = safe_malloc(sizeof(float)*n*m);

	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			a[i*m+j] = (float)rand()/(float)RAND_MAX;

	*matrix = a;
}

// Print a 2D Matrix of size NxM
void print_2d_matrix(float* matrix, int n, int m) {
	int i, j;
	float* a = matrix;

	printf("[\n");
	for(i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf("%f ", a[i*m+j]);
		}
		printf("\n");
	}
	printf("]\n");
}

// Calculate the average of an array of floats.
// Note that the first element is excluded.
float calculate_average(float* array, int repeat) {
	float sum = calculate_sum(array, repeat);

	float average = (sum - array[0]) / (double)(repeat-1);

	return average;
}

// Calculate the summation of an array of floats.
float calculate_sum(float* array, int repeat) {
	int i;
	float sum = 0;

	for (i = 0; i < repeat; i++) {
		sum += array[i];
	}

	return sum;
}

// Convert time from micro-seconds to seconds
float convert_to_sec(long long time_usec) {
	return ((float)time_usec * 1.0e-06);
}

// Get the next integer multiple of a number
int get_next_int_multiple(int number, int multiple) {
	if (multiple == 0) {
		return number;
	}

	int remainder = number % multiple;
	if (remainder == 0) {
		return number;
	}

	return number + multiple - remainder;
}

