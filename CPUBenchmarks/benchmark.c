#include "benchmark.h"

// Handle errors from the PAPI library
void handle_papi_error(int papi_retval)
{
	fprintf(stderr, "Error: PAPI error %d: %s\n", papi_retval,
	PAPI_strerror(papi_retval));
	exit(-1);
}

// Initialize the PAPI library
void initialize_papi()
{
	// Initialize the PAPI library and get the number of counters available
	int papi_retval = PAPI_num_counters();
	if (papi_retval < PAPI_OK)
		handle_papi_error(papi_retval);

	if (papi_retval < NUM_COUNTERS) {
		fprintf(stderr, "Error: not enough hardware counters. "
				"This system has only %d available.\n",
				papi_retval);
	}

		papi_retval = PAPI_thread_init(pthread_self);
		if (papi_retval != PAPI_OK)
			handle_papi_error(papi_retval);

		unsigned long int thread_id;
		thread_id = PAPI_thread_id();
		if (thread_id == (unsigned long int)-1)
			handle_papi_error(1);
}

// Get benchmark options from the commandline arguments
void get_benchmark_options(benchmark_options *bmo, int argc,
		char *argv[], int threads, int benchmark) {
	bmo->threads = threads;
	bmo->benchmark = benchmark;
	bmo->repeat = 0;
	bmo->details = 0;
	bmo->dim = 0;
	bmo->ratio = 0;

	if (threads < 0) {
		fprintf(stderr, "Error: invalid number of threads (<0)!\n");
		exit(-1);
	}

	if (benchmark >= NUM_MICROBENCHMARKS) {
		fprintf(stderr, "Error: invalid benchmark "
			"(choose one < %d)!\n", NUM_MICROBENCHMARKS);
		exit(-1);
	}

	int benchmark_args[NUM_MICROBENCHMARKS] = {
		3,
		3,
		3,
		3,
		3,
		3,
		2,
		2,
		2,
		2
	};

	int required_args = benchmark_args[benchmark];
	if (argc < required_args) {
		fprintf(stderr, "Error: missing arguments! (<%d)\n", required_args);
		exit(-1);
	}

	bmo->repeat = atoi(argv[0]);
	if (bmo->repeat < 2) {
		fprintf(stderr, "Error: invalid number of repetitions (<2)!\n");
		exit(-1);
	}

	bmo->details = atoi(argv[1]);

	if (benchmark == BM_MatrixMultiply ||
			benchmark == BM_PolynomialEvaluation ||
			benchmark == BM_StoreSimple ||
			benchmark == BM_StoreUnroll ||
			benchmark == BM_StoreLongUnroll) {
		bmo->dim = atoi(argv[2]);
		if (bmo->dim < 1) {
			fprintf(stderr, "Error: invalid dimensions (<1)!\n");
			exit(-1);
		}
		else if (benchmark == BM_StoreUnroll && bmo->dim % 4 != 0) {
			fprintf(stderr, "Error: invalid size (must be divisable by 4)!\n");
			exit(-1);
		}
		else if (benchmark == BM_StoreLongUnroll && bmo->dim % 8 != 0) {
			fprintf(stderr, "Error: invalid size (must be divisable by 8)!""\n");
			exit(-1);
		}
	}
	else if (benchmark == BM_Roofline) {
		bmo->ratio = atoi(argv[2]);
		if (bmo->ratio < 0) {
			fprintf(stderr, "Error: invalid ratio (<0)!\n");
			exit(-1);
		}
	}
}

void prepare_benchmark_data(benchmark_options *bmo) {
	const char* const microbenchmark_description[NUM_MICROBENCHMARKS] = {
		"Matrix Multiply",
		"Polynomial Evaluation",
		"Roofline",
		"Store Simple",
		"Store Unroll",
		"Store Long Unroll",
		"Floating Point Add",
		"Floating Point Fused Multiply Add",
		"SIMD 128bit Floating Point Add",
		"Mixed SIMD and Floating Point Fused Multiply Add"
	};

	if (bmo->benchmark == BM_MatrixMultiply) {
		if (bmo->details != DL_min && bmo->details != DL_threads)
			printf("Generating matrices (dimension = %d)\n", bmo->dim);
		prepare_sgemm_data(bmo->dim, bmo->threads);
	}
	else if (bmo->benchmark == BM_PolynomialEvaluation) {
		if (bmo->details != DL_min && bmo->details != DL_threads)
			printf("Generating polynomial (dimension = %d)\n", bmo->dim);
		prepare_poly_eval_data(bmo->dim, bmo->threads);
	}
	else if (bmo->benchmark == BM_Roofline) {
		prepare_roofline_data(bmo->ratio, bmo->threads);
	}
	else if (bmo->benchmark == BM_StoreSimple) {
		prepare_store_simple_data(bmo->dim, bmo->threads);
	}
	else if (bmo->benchmark == BM_StoreUnroll) {
		prepare_store_unroll_data(bmo->dim, bmo->threads);
	}
	else if (bmo->benchmark == BM_StoreLongUnroll) {
		prepare_store_long_unroll_data(bmo->dim, bmo->threads);
	}
	else if (bmo->benchmark == BM_SIMD) {
		prepare_neon_data(NUM_LOOPS);
	}

	if (bmo->details != DL_min && bmo->details != DL_threads)
		printf("Starting %s benchmark\n",
			microbenchmark_description[bmo->benchmark]);
}

void delete_benchmark_data(benchmark_options *bmo) {
	if (bmo->benchmark == BM_MatrixMultiply)
		delete_sgemm_data();
	else if (bmo->benchmark == BM_PolynomialEvaluation)
		delete_poly_eval_data();
	else if (bmo->benchmark == BM_Roofline)
		delete_roofline_data();
	else if (bmo->benchmark == BM_StoreSimple)
		delete_store_simple_data();
	else if (bmo->benchmark == BM_StoreUnroll)
		delete_store_unroll_data();
	else if (bmo->benchmark == BM_StoreLongUnroll)
		delete_store_long_unroll_data();
}

// Run benchmarks in parallel
void parallel_benchmark_microbenchmark(benchmark_options *bmo) {
	omp_set_dynamic(0);
	omp_set_num_threads(bmo->threads);

	initialize_papi();

	void (*microbenchmark[NUM_MICROBENCHMARKS])(int) = {
		microbenchmark_sgemm,
		microbenchmark_poly_eval,
		microbenchmark_roofline,
		microbenchmark_store_simple,
		microbenchmark_store_unroll,
		microbenchmark_store_long_unroll,
		microbenchmark_fadds,
		microbenchmark_fmacs,
		microbenchmark_neon,
		microbenchmark_neon_fmacs
	};

	thread_results_summary results;
	results.intensity = safe_malloc(sizeof(float)*bmo->threads);
	results.mflops = safe_malloc(sizeof(float)*bmo->threads);
	results.mem_bw = safe_malloc(sizeof(float)*bmo->threads);

	#pragma omp parallel
	{
		int thread_num = omp_get_thread_num();
		PAPI_register_thread();

		// Start counting events
		int events[NUM_COUNTERS] = {PAPI_FP_INS, PAPI_LD_INS, PAPI_SR_INS,
			PAPI_VEC_INS, PAPI_L2_DCM};
		int papi_retval = PAPI_start_counters(events, NUM_COUNTERS);
		if (papi_retval != PAPI_OK)
			handle_papi_error(papi_retval);

		if (bmo->benchmark == BM_StoreLongUnroll)
			prepare_store_long_unroll_thread(thread_num);
		else if (bmo->benchmark == BM_Roofline)
			prepare_roofline_thread(thread_num);
		else if (bmo->benchmark == BM_StoreUnroll)
			prepare_store_unroll_thread(thread_num);
		else if (bmo->benchmark == BM_StoreSimple)
			prepare_store_simple_thread(thread_num);
		else if (bmo->benchmark == BM_PolynomialEvaluation)
			prepare_poly_eval_thread(thread_num);
		else if (bmo->benchmark == BM_MatrixMultiply)
			prepare_sgemm_thread(thread_num);

		benchmark_microbenchmark(bmo->repeat, bmo->details, bmo->benchmark,
				microbenchmark[bmo->benchmark], &results);

		if (bmo->benchmark == BM_StoreLongUnroll)
			delete_store_long_unroll_thread(thread_num);
		else if (bmo->benchmark == BM_Roofline)
			delete_roofline_thread(thread_num);
		else if (bmo->benchmark == BM_StoreUnroll)
			delete_store_unroll_thread(thread_num);
		else if (bmo->benchmark == BM_StoreSimple)
			delete_store_simple_thread(thread_num);
		else if (bmo->benchmark == BM_PolynomialEvaluation)
			delete_poly_eval_thread(thread_num);
		else if (bmo->benchmark == BM_MatrixMultiply)
			delete_sgemm_thread(thread_num);

		PAPI_unregister_thread();
	}

	if (bmo->details == DL_threads) {
		float sum_mflops = calculate_sum(results.mflops, bmo->threads);
		float sum_mem_bw = calculate_sum(results.mem_bw, bmo->threads);
		float sum_intensity = sum_mflops / sum_mem_bw;

		printf("%f, %f, %f\n",
			sum_intensity, sum_mflops,
			sum_mem_bw);
	}

	free(results.intensity);
	free(results.mflops);
	free(results.mem_bw);
}

// Benchmark function
void benchmark_microbenchmark(int repeat, int details, int benchmark,
				void (*microbenchmark)(int), thread_results_summary *results) {
	int run, papi_retval;
	int thread_num = omp_get_thread_num();
	long long real_time[2], proc_time[2];
	long long values[NUM_COUNTERS];

	float ops_ips_ratio, mem_ratio;

	if (benchmark == BM_FPFusedMultiplyAdd) {
		ops_ips_ratio = 2.0f;
		mem_ratio = 1.0f;
	}
	else if (benchmark == BM_SIMD) {
		ops_ips_ratio = 4.0f;
		mem_ratio = 1.0f;
	}
	else if (benchmark == BM_Roofline) {
		ops_ips_ratio = 4.0f;
		mem_ratio = 2.0f;
	}
	else if (benchmark == BM_StoreLongUnroll) {
		ops_ips_ratio = 1.0f;
		mem_ratio = 2.0f;
	}
	else {
		ops_ips_ratio = 1.0f;
		mem_ratio = 1.0f;
	}

	float *run_real_time = safe_malloc(sizeof(float)*repeat);
	float *run_proc_time = safe_malloc(sizeof(float)*repeat);
	float *run_mflips = safe_malloc(sizeof(float)*repeat);
	float *run_mem_ld_bw = safe_malloc(sizeof(float)*repeat);
	float *run_mem_st_bw = safe_malloc(sizeof(float)*repeat);
	float *run_mvips = safe_malloc(sizeof(float)*repeat);
	float *run_l2_dcm_per_sec = safe_malloc(sizeof(float)*repeat);

	if (details == DL_all || details == DL_run) {
		printf("Run #, Real Time, Proc Time, FLIPS, MFLIPS, "
			"VIPS, MVIPS, OPs to IPs Ratio, MEM Ratio, LDs, "
			"Read MBps, STs, Write MBPs, L2 DCM, L2 DCM/Sec\n");
	}

	for (run = 0; run < repeat; run++) {
		real_time[0] = PAPI_get_real_usec();
		proc_time[0] = PAPI_get_virt_usec();

		papi_retval = PAPI_read_counters(values, NUM_COUNTERS);

		// Run the specified microbenchmark
		microbenchmark(thread_num);

		papi_retval = PAPI_read_counters(values, NUM_COUNTERS);

		real_time[1] = PAPI_get_real_usec();
		proc_time[1] = PAPI_get_virt_usec();

		run_real_time[run] = convert_to_sec(real_time[1]
				- real_time[0]);
		run_proc_time[run] = convert_to_sec(proc_time[1]
				- proc_time[0]);
		run_mflips[run] = (float)values[0] / run_proc_time[run]
			* 1.0e-06;
		run_mem_ld_bw[run] = (float)(values[1] * sizeof(float))
			/ (run_proc_time[run] * 1024 * 1024);
		run_mem_st_bw[run] = (float)(values[2] * sizeof(float))
			/ (run_proc_time[run] * 1024 * 1024);
		run_mvips[run] = (float)values[3] / run_proc_time[run]
			* 1.0e-06;
		run_l2_dcm_per_sec[run] = (float)values[4] / run_proc_time[run];

		if (papi_retval != PAPI_OK)
			handle_papi_error(papi_retval);

		if (details == DL_all || details == DL_run) {
			printf("%d, %f, %f, %lld, %f, %lld, %f, %f, %f, %lld, "
					"%f, %lld, %f, %lld, %f\n", run,
					run_real_time[run], run_proc_time[run],
					values[0], run_mflips[run],
					values[3], run_mvips[run],
					ops_ips_ratio, mem_ratio,
					values[1], run_mem_ld_bw[run],
					values[2], run_mem_st_bw[run],
					values[4], run_l2_dcm_per_sec[run]);
		}
	}

	float average_run_real_time = calculate_average(run_real_time, repeat);
	float average_run_proc_time = calculate_average(run_proc_time, repeat);
	float average_run_mflips = calculate_average(run_mflips, repeat);
	float average_run_mem_ld_bw = calculate_average(run_mem_ld_bw, repeat)
		* mem_ratio;
	float average_run_mem_st_bw = calculate_average(run_mem_st_bw, repeat)
		* mem_ratio;
	float average_run_mvips = calculate_average(run_mvips, repeat);
	float average_run_mflops = ((average_run_mflips > average_run_mvips) ?
					average_run_mflips : average_run_mvips)
					* ops_ips_ratio;
	float average_run_mem_bw = (average_run_mem_ld_bw +
					average_run_mem_st_bw);
	float average_run_intensity = (average_run_mflops / average_run_mem_bw)
					* 1024 * 1024 * 1.0e-06;
	float average_run_l2_dcm_per_sec = calculate_average(run_l2_dcm_per_sec,
			repeat);

	results->intensity[thread_num] = average_run_intensity;
	results->mflops[thread_num] = average_run_mflops;
	results->mem_bw[thread_num] = average_run_mem_bw;

	if (details == DL_all || details == DL_avg) {
		printf("Average over %d runs:\n", repeat - 1);
		printf("    %f real seconds, %f proc seconds, %f MFLIPS, "
				"%f MVIPS, %f OPs to IPs Ratio\n",
				average_run_real_time, average_run_proc_time,
				average_run_mflips, average_run_mvips,
				ops_ips_ratio);
		printf("    %f Mem Ratio, %f Read MBps, %f Write MBps\n",
				mem_ratio, average_run_mem_ld_bw,
				average_run_mem_st_bw);
		printf("    %f L2 DCM/Sec\n", average_run_l2_dcm_per_sec);
		printf("    %f Intensity, %f MFLOPs, %f Mem MBps\n",
				average_run_intensity, average_run_mflops,
				average_run_mem_bw);
	}
	else if (details == DL_min) {
		printf("%f, %f, %f\n",
			average_run_intensity, average_run_mflops,
			average_run_mem_bw);
	}

	papi_retval = PAPI_stop_counters(values, NUM_COUNTERS);
	if (papi_retval != PAPI_OK)
		handle_papi_error(papi_retval);

	free(run_real_time);
	free(run_proc_time);
	free(run_mflips);
	free(run_mem_ld_bw);
	free(run_mem_st_bw);
	free(run_mvips);
}

