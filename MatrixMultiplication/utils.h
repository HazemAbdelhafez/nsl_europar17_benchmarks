/*
 * utils.h
 *
 *  Created on: Dec 14, 2016
 *      Author: hazem
 */
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>
#include "papi.h"
#include <cblas.h>
#include <pthread.h>
#include <sys/mman.h>
#ifndef UTILS_H_
#define UTILS_H_

float convert_to_sec(long long time_usec);
float calculate_average(float* array, int repeat);
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A);
void fill_2d_matrix(float* matrix, int n, int m, float value);
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A);
#endif /* UTILS_H_ */
