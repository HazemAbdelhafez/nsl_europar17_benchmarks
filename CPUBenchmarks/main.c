#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "benchmark.h"

#define NUM_THREADS 4

int main(int argc, char* argv[]) {
	if (argc < 3)
	{
		printf("Microbenchmarks 6 to %d:\n", NUM_MICROBENCHMARKS - 1);
		printf("Usage:   %s benchmark threads repeat details\n", argv[0]);
		printf("Usage:   %s 6 4 10 0\n", argv[0]);
		printf("Or\n");
		printf("Store Long Unroll:\n");
		printf("Usage:   %s 5 threads repeat details size\n", argv[0]);
		printf("Example: %s 5 4 10 0 4000\n", argv[0]);
		printf("Or\n");
		printf("Store Unroll:\n");
		printf("Usage:   %s 4 threads repeat details size\n", argv[0]);
		printf("Example: %s 4 4 10 0 4000\n", argv[0]);
		printf("Or\n");
		printf("Store Simple:\n");
		printf("Usage:   %s 3 threads repeat details size\n", argv[0]);
		printf("Example: %s 3 4 10 0 4000\n", argv[0]);
		printf("Or\n");
		printf("Roofline:\n");
		printf("Usage:   %s 2 threads repeat details ratio\n", argv[0]);
		printf("Example: %s 2 4 10 0 10\n", argv[0]);
		printf("Or\n");
		printf("Polynomial Evaluation:\n");
		printf("Usage:   %s 1 threads repeat details degree\n", argv[0]);
		printf("Example: %s 1 4 10 0 2048\n", argv[0]);
		printf("Or\n");
		printf("Matrix Multiply:\n");
		printf("Usage:   %s 0 threads repeat details dimension\n", argv[0]);
		printf("Example: %s 0 3 10 0 2048\n", argv[0]);
		exit(-1);
	}

	srand(time(NULL));

	benchmark_options bmo;
	get_benchmark_options(&bmo, argc - 3, &argv[3], atoi(argv[2]), atoi(argv[1]));

	prepare_benchmark_data(&bmo);

	// Start the benchmark
	parallel_benchmark_microbenchmark(&bmo);

	delete_benchmark_data(&bmo);

	return 0;
}

