#ifndef MB_COMMON_H
#define MB_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_LOOPS 1000000

// Allocate array using malloc and check if successful
void *safe_malloc(size_t size);

// Measure the current time in seconds
double get_time_seconds();

// Fill 2D Matrix of size NxM with a provided value
void fill_2d_matrix(float** matrix, int n, int m, float value);

// Create a random 2D Matrix of size NxM
void random_2d_matrix(float** matrix, int n, int m);

// Print a 2D Matrix of size NxM
void print_2d_matrix(float* matrix, int n, int m);

// Calculate the average of an array of floats.
// Note that the first element is excluded.
float calculate_average(float* array, int repeat);

// Calculate the summation of an array of floats.
float calculate_sum(float* array, int repeat);

// Convert time from micro-seconds to seconds
float convert_to_sec(long long time_usec);

// Get the next integer multiple of a number
int get_next_int_multiple(int number, int multiple);

#endif

