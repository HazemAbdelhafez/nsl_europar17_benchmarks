#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "benchmark.h"
#include "mb_roofline.h"

#define REPEAT 5
#define RATIO_START 0
#define RATIO_STEP 2
#define RATIO_STOP 32768

int main(int argc, char* argv[]) {
	srand(time(NULL));
	benchmark_options bmo;

	if (argc < 5) {
		printf("Roofline benchmark\n");
		printf("Usage:    %s threads details fp_loops store_size\n", argv[0]);
		printf("Usage:    %s 4 4 4000000 4000000\n", argv[0]);
		exit(-1);
	}

	bmo.threads = atoi(argv[1]);
	bmo.benchmark = BM_Roofline;
	bmo.repeat = REPEAT;
	bmo.details = atoi(argv[2]);
	bmo.ratio = RATIO_START;
	int fp_loops = atoi(argv[3]);
	int store_size = atoi(argv[4]);

	prepare_roofline_data_size(bmo.ratio, bmo.threads, fp_loops, store_size);
	printf("0, ");
	parallel_benchmark_microbenchmark(&bmo);

	for(bmo.ratio = 1; bmo.ratio < RATIO_STOP; bmo.ratio *= RATIO_STEP) {
		ROOFLINE_DATA.fp_mem_ratio = bmo.ratio;
		printf("%d, ", bmo.ratio);
		// Start the benchmark
		parallel_benchmark_microbenchmark(&bmo);
	}

	delete_roofline_data();
	return 0;
}

