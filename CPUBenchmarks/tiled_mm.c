#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "benchmark.h"
#include "mb_tiled_mm.h"

#define REPEAT 5
#define RATIO_START 0
#define RATIO_STEP 2
#define RATIO_STOP 32768

int main(int argc, char* argv[]) {
	srand(time(NULL));
	benchmark_options bmo;

	if (argc < 5) {
		printf("Tiled Matrix Multiply benchmark\n");
		printf("Usage:    %s threads details dimension block_size\n", argv[0]);
		printf("Usage:    %s 4 4 1024 16\n", argv[0]);
		exit(-1);
	}

	bmo.threads = atoi(argv[1]);
	bmo.benchmark = BM_Tiled_MM;
	bmo.repeat = REPEAT;
	bmo.details = atoi(argv[2]);
	bmo.ratio = RATIO_START;
	bmo.dim = atoi(argv[3]);
	bmo.block_size = atoi(argv[4]);

	prepare_tiled_mm_data(bmo.dim, bmo.block_size, bmo.threads);
	parallel_benchmark_microbenchmark(&bmo);

	/* for(bmo.ratio = 1; bmo.ratio < RATIO_STOP; bmo.ratio *= RATIO_STEP) { */
	/* 	ROOFLINE_DATA.fp_mem_ratio = bmo.ratio; */
	/* 	printf("%d, ", bmo.ratio); */
	/* 	// Start the benchmark */
	/* 	parallel_benchmark_microbenchmark(&bmo); */
	/* } */

	delete_tiled_mm_data();
	return 0;
}

