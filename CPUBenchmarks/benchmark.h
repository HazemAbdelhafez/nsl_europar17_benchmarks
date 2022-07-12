#ifndef MB_BENCHMARK_H
#define MB_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <papi.h>
#include "common.h"
#include "mb_fadds.h"
#include "mb_fmacs.h"
#include "mb_neon.h"
#include "mb_neon_fmacs.h"
#include "mb_polyeval.h"
#include "mb_roofline.h"
#include "mb_sgemm.h"
#include "mb_store_simple.h"
#include "mb_store_unroll.h"
#include "mb_store_long_unroll.h"

#define NUM_COUNTERS 5
#define NUM_MICROBENCHMARKS 10

typedef struct {
	int threads;
	int benchmark;
	int repeat;
	int details;
	int dim;
	int ratio;
} benchmark_options;

typedef struct {
	int threads;
	float *intensity;
	float *mflops;
	float *mem_bw;
} thread_results_summary;

enum benchmarktype {
	BM_MatrixMultiply,
	BM_PolynomialEvaluation,
	BM_Roofline,
	BM_StoreSimple,
	BM_StoreUnroll,
	BM_StoreLongUnroll,
	BM_FPAdd,
	BM_FPFusedMultiplyAdd,
	BM_SIMD,
	BM_MixedSIMDFusedMultiplyAdd
};

enum detaillevel {
	DL_avg,
	DL_run,
	DL_all,
	DL_min,
	DL_threads
};

// Handle errors from the PAPI library
void handle_papi_error(int papi_retval);

// Initialize the PAPI library
void initialize_papi();

// Get benchmark options from the commandline arguments
void get_benchmark_options(benchmark_options *bmo, int argc,
		char *argv[], int threads, int benchmark);

void prepare_benchmark_data(benchmark_options *bmo);

void delete_benchmark_data(benchmark_options *bmo);

// Run benchmarks in parallel
void parallel_benchmark_microbenchmark(benchmark_options *bmo);

// Benchmark function
void benchmark_microbenchmark(int repeat, int details, int benchmark,
				void (*microbenchmark)(int), thread_results_summary *results);

#endif

