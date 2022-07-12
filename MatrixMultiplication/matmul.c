/*
 * matmul.c
 *
 *  Created on: Dec 14, 2016
 *      Author: hazem
 */
#include "matmul.h"

// Matrix multiplication GPU kernels
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=k,ldb=n,ldc=n;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Create a handle for CUBLAS;
     cublasHandle_t handle;
     cublasCreate(&handle);

     // Do the actual multiplication
     // Use the number of columns assigned to the GPU instead of n, we use B*A = C because cuBLAS will
     // automatically transpose A and B to be Bt*At = (AB)T = (Ct)t. This happens because default behavior of cuBLAS is A*B = Ct
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);

     // Destroy the handle
     cublasDestroy(handle);
}
void gpu_blas_split(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=k,ldb=n,ldc=n;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Create a handle for CUBLAS;
     cublasHandle_t handle;
     cublasCreate(&handle);

     // Do the actual multiplication
     // Use the number of columns assigned to the GPU instead of n, we use B*A = C because cuBLAS will
     // automatically transpose A and B to be Bt*At = (AB)T = (Ct)t. This happens because default behavoir of cuBLAS is A*B = Ct
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, gpuCols, m, k, alpha, &B[dim-gpuCols], ldb, A, lda, beta, &C[dim-gpuCols], ldc);

     // Destroy the handle
     cublasDestroy(handle);
}
void gpu_blas_split_with_stream(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=k,ldb=n,ldc=n;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Create a handle for CUBLAS;
     cublasHandle_t handle;
     cublasCreate(&handle);
     cublasSetStream(handle, stream1);
     // Do the actual multiplication
     // Use the number of columns assigned to the GPU instead of n, we use B*A = C because cuBLAS will
     // automatically transpose A and B to be Bt*At = (AB)T = (Ct)t. This happens because default behavoir of cuBLAS is A*B = Ct
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, gpuCols, m, k, alpha, &B[dim-gpuCols], ldb, A, lda, beta, &C[dim-gpuCols], ldc);

     // Destroy the handle
     cublasDestroy(handle);
}

// Matrix multiplication CPU kernels
void cpu_blas_split(float *A, float *B, float *C, const int m, const int k, const int n) {
    int lda=k,ldb=n,ldc=n;
    int alf = 1;
    int bet = 0;
 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
 			m,		 		// M
 			cpuCols,    	// N
 			k,          		// K
 			alf,							// Alpha
 			A, 	   		// Matrix A
 			lda,		 		// LDA
 			B,  			// Matrix B
 			ldb,	 			// LDB
 			bet,							// Beta
 			C,			// Matrix C
 			ldc);				// LDC
}
void cpu_blas_mmul(float *A, float *B, float *C, const int m, const int k, const int n) {
    int lda=k,ldb=n,ldc=n;
    int alf = 1;
    int bet = 0;
 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
 			m,		 			// M
 			n,    				// N
 			k,          		// K
 			alf,				// Alpha
 			A, 	   				// Matrix A
 			lda,		 		// LDA
 			B,  				// Matrix B
 			ldb,	 			// LDB
 			bet,				// Beta
 			C,					// Matrix C
 			ldc);				// LDC
}

// GPU Benchmarks
void gpuMatMul(){

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate result matrixC on CPU memory just for printing (optional)
	float *cmatC = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float *gmatA, *gmatB, *gmatC;
	cudaMalloc((void**) &gmatA, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc((void**) &gmatB, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc((void**) &gmatC, nr_rows_C * nr_cols_C * sizeof(float));

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(gmatA, nr_rows_A, nr_cols_A);
	GPU_fill_rand(gmatB, nr_rows_B, nr_cols_B);

    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
	for (int i=0; i < repeats; i++){

		real_time[0] = PAPI_get_real_usec();

		// Multiply A and B on GPU
		gpu_blas_mmul(gmatA, gmatB, gmatC, nr_rows_A, nr_cols_A, nr_cols_B);

		 // Copy (and print) the result on host memory
		cudaMemcpy(cmatC, gmatC,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);

	}

	float averageTime = calculate_average(run_real_time, repeats);
	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(gmatC, nr_rows_C, nr_cols_C);
    }

	// Free GPU memory
	cudaFree(gmatA);
	cudaFree(gmatB);
	cudaFree(gmatC);
    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);

}
void gpuMatMulWithUM(){

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate 3 arrays on GPU
    float *gmatA, *gmatB, *gmatC;
    cudaMallocManaged((void**) &gmatA, nr_rows_A * nr_cols_A * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gmatB, nr_rows_B * nr_cols_B * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gmatC, nr_rows_C * nr_cols_C * sizeof(float),cudaMemAttachGlobal);

    // Fill the arrays A and B on GPU with random numbers
    fill_2d_matrix(gmatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(gmatB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(gmatC, nr_rows_C, nr_cols_C,0.0);

    // Debugging
    if (showDetails == 1){
    	printf("Matrices before multiplication\n");
		printf("A = \n");
		print_matrix(gmatA, nr_rows_A, nr_cols_A);
		printf("B = \n");
		print_matrix(gmatB, nr_rows_B, nr_cols_B);
		printf("C = \n");
		print_matrix(gmatC, nr_rows_C, nr_cols_C);
    }
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);

    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		real_time[0] = PAPI_get_real_usec();

		// Multiply A and B on GPU
		gpu_blas_mmul(gmatA, gmatB, gmatC, nr_rows_A, nr_cols_A, nr_cols_B);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);

	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}
    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(gmatC, nr_rows_C, nr_cols_C);
    }
     // Free GPU memory
     cudaFree(gmatA);
     cudaFree(gmatB);
     cudaFree(gmatC);

     // Output the gflops and time passed
     printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);

}
void gpuMatMulWithMemCpy(){

	// For simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate arrays on CPU memory
	float *cmatA = (float*) malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *cmatB = (float*) malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *cmatC = (float*) malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
    float *gmatA, *gmatB, *gmatC;
    cudaMalloc((void**) &gmatA, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc((void**) &gmatB, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc((void**) &gmatC, nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the arrays A and B on GPU with random numbers
    fill_2d_matrix(cmatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(cmatB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(cmatC, nr_rows_C, nr_cols_C,0.0);

    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Copy the arrays to the GPU memory
		cudaMemcpy(gmatA,cmatA, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gmatB,cmatB, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gmatC,cmatC, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyHostToDevice);

		// Multiply A and B on GPU
		gpu_blas_mmul(gmatA, gmatB, gmatC, nr_rows_A, nr_cols_A, nr_cols_B);

		 // Copy (and print) the result on host memory
		cudaMemcpy(cmatC,gmatC,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(cmatC, nr_rows_C, nr_cols_C);
    }

     // Free GPU memory
    cudaFree(gmatA);
    cudaFree(gmatB);
    cudaFree(gmatC);

     // Free CPU memory
	free(cmatA);
	free(cmatB);
	free(cmatC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}

// CPU Benchmarks
void cpuMatMul(){

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate 3 arrays on cPU
    float *gmatA, *gmatB, *gmatC;
    gmatA = (float*) malloc(nr_rows_A * nr_cols_A * sizeof(float));
    gmatB = (float*) malloc(nr_rows_B * nr_cols_B * sizeof(float));
    gmatC = (float*) malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the arrays A and B on GPU with random numbers
    fill_2d_matrix(gmatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(gmatB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(gmatC, nr_rows_C, nr_cols_C,0.0);

    // Debugging
    if (showDetails == 1){
    	printf("Matrices before multiplication\n");
		printf("A = \n");
		print_matrix(gmatA, nr_rows_A, nr_cols_A);
		printf("B = \n");
		print_matrix(gmatB, nr_rows_B, nr_cols_B);
		printf("C = \n");
		print_matrix(gmatC, nr_rows_C, nr_cols_C);
    }

    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);

    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Multiply A and B on GPU
		cpu_blas_mmul(gmatA, gmatB, gmatC, nr_rows_A, nr_cols_A, nr_cols_B);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);

	}

	float averageTime = calculate_average(run_real_time, repeats);
	if (showDetails){
		printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
		printf("Runtime: %f Seconds\n", averageTime);
	}
    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(gmatC, nr_rows_C, nr_cols_C);
    }
     // Free GPU memory
     free(gmatA);
     free(gmatB);
     free(gmatC);

     // Output the gflops and time passed
     printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);

}
void cpuMatMulWithUM(){

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate 3 arrays on GPU
    float *gmatA, *gmatB, *gmatC;
    cudaMallocManaged((void**) &gmatA, nr_rows_A * nr_cols_A * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gmatB, nr_rows_B * nr_cols_B * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gmatC, nr_rows_C * nr_cols_C * sizeof(float),cudaMemAttachGlobal);

    // Fill the arrays A and B on GPU with random numbers
    fill_2d_matrix(gmatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(gmatB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(gmatC, nr_rows_C, nr_cols_C,0.0);

    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);

    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Multiply A and B on GPU
		cpu_blas_mmul(gmatA, gmatB, gmatC, nr_rows_A, nr_cols_A, nr_cols_B);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);

	}
 	// Copy (and print) the result on host memory
 	// cudaMemcpy(h_C,gmatC,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
 	// printf("C = \n");
 	// print_matrix(cmatC, nr_rows_C, nr_cols_C);
	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

     // Free GPU memory
     cudaFree(gmatA);
     cudaFree(gmatB);
     cudaFree(gmatC);

     // Output the gflops and time passed
     printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}
void cpuMatMultWithHostAlloc(){

	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate shared matrices
    cudaMallocHost((void**) &matA, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMallocHost((void**) &matB, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMallocHost((void**) &matC, nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the matrices
    fill_2d_matrix(matA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(matB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(matC, nr_rows_C, nr_cols_C,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Multiply A and B on GPU
		cpu_blas_mmul(matA, matB, matC, nr_rows_A, nr_cols_A, nr_cols_B);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(matC, nr_rows_C, nr_cols_C);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}

// Heterogeneous Benchmarks
void heterogeneousWithHostAlloc(){

	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate shared matrices
    cudaMallocHost((void**) &matA, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMallocHost((void**) &matB, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMallocHost((void**) &matC, nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the matrices
    fill_2d_matrix(matA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(matB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(matC, nr_rows_C, nr_cols_C,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Launch the CPU and GPU threads
		int cpuThreadError = pthread_create(&(tid[0]), NULL, &cpuThread, NULL);
		int gpuThreadError = pthread_create(&(tid[1]), NULL, &gpuThread, NULL);
		pthread_join(tid[0], NULL);
		pthread_join(tid[1], NULL);
		cudaDeviceSynchronize();

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(matC, nr_rows_C, nr_cols_C);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}
void heterogeneousWithZeroMemCpy(){

	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;
	cudaSetDeviceFlags(cudaDeviceMapHost);

	// Allocate shared matrices
    cudaHostAlloc((void**) &matA, nr_rows_A * nr_cols_A * sizeof(float),cudaHostAllocWriteCombined);
    cudaHostAlloc((void**) &matB, nr_rows_B * nr_cols_B * sizeof(float),cudaHostAllocWriteCombined);
    cudaHostAlloc((void**) &matC, nr_rows_C * nr_cols_C * sizeof(float),cudaHostAllocWriteCombined);

    // Get device pointers
//    cudaHostGetDevicePointer((void **)&mappedMatA,  (void *) matA , 0);
//    cudaHostGetDevicePointer((void **)&mappedMatB,  (void *) matB , 0);
//    cudaHostGetDevicePointer((void **)&mappedMatC,  (void *) matC , 0);

    // Fill the matrices
    fill_2d_matrix(matA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(matB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(matC, nr_rows_C, nr_cols_C,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Launch the CPU and GPU threads
		int cpuThreadError = pthread_create(&(tid[0]), NULL, &cpuThread, NULL);
		int gpuThreadError = pthread_create(&(tid[1]), NULL, &gpuThread, NULL);
		pthread_join(tid[0], NULL);
		pthread_join(tid[1], NULL);
		cudaDeviceSynchronize();

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(matC, nr_rows_C, nr_cols_C);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}
void heterogeneousWithAsyncMemCpy(){

	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate 3 arrays on CPU
    matA = (float*) malloc(nr_rows_A * nr_cols_A * sizeof(float));
    matB = (float*) malloc(nr_rows_B * nr_cols_B * sizeof(float));
    matC = (float*) malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on the GPU
    cudaMalloc((void**) &mappedMatA, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc((void**) &mappedMatB, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc((void**) &mappedMatC, nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the matrices
    fill_2d_matrix(matA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(matB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(matC, nr_rows_C, nr_cols_C,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Create stream for handling Async memory copy
		cudaStreamCreate(&stream1);

		// Copy the arrays to the GPU memory
		cudaMemcpyAsync(mappedMatA,matA, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(mappedMatB,matB, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice, stream1);

		// Launch the CPU and GPU threads
		int cpuThreadError = pthread_create(&(tid[0]), NULL, &cpuThread, NULL);
		int gpuThreadError = pthread_create(&(tid[1]), NULL, &gpuThreadAsyncMemCpy, NULL);
		pthread_join(tid[0], NULL);
		pthread_join(tid[1], NULL);
		cudaDeviceSynchronize();

	 	// Copy the result on host memory
    	cudaMemcpy(matC,mappedMatC,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(matC, nr_rows_C, nr_cols_C);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}
void heterogeneousWithSyncMemCpy(){

	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate 3 arrays on CPU
    matA = (float*) malloc(nr_rows_A * nr_cols_A * sizeof(float));
    matB = (float*) malloc(nr_rows_B * nr_cols_B * sizeof(float));
    matC = (float*) malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on the GPU
    cudaMalloc((void**) &mappedMatA, nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc((void**) &mappedMatB, nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc((void**) &mappedMatC, nr_rows_C * nr_cols_C * sizeof(float));

    // Fill the matrices
    fill_2d_matrix(matA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(matB, nr_rows_B, nr_cols_B,1.0);
    fill_2d_matrix(matC, nr_rows_C, nr_cols_C,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Copy the arrays to the GPU memory
		cudaMemcpy(matA, cmatA, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(matB, cmatB, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);

		// Launch the CPU and GPU threads
		int cpuThreadError = pthread_create(&(tid[0]), NULL, &cpuThread, NULL);
		int gpuThreadError = pthread_create(&(tid[1]), NULL, &gpuThreadSyncMemCpy, NULL);
		pthread_join(tid[0], NULL);
		pthread_join(tid[1], NULL);
		cudaDeviceSynchronize();

	 	// Copy the result on host memory
    	cudaMemcpy(mappedMatC, matC, nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(matC, nr_rows_C, nr_cols_C);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);
}

void heterogeneousWithSplitMem(){
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

	// Allocate CPU portion of the matrix
    cudaMallocManaged((void**) &cpuMatA, nr_rows_A * nr_cols_A * sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void**) &cpuMatB, nr_rows_B * cpuCols * sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void**) &cpuMatC, nr_rows_C * cpuCols * sizeof(float),cudaMemAttachHost);

    // Allocate the GPU portion of the matrix
    cudaMallocManaged((void**) &gpuMatA, nr_rows_A * nr_cols_A * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gpuMatB, nr_rows_B * gpuCols * sizeof(float),cudaMemAttachGlobal);
    cudaMallocManaged((void**) &gpuMatC, nr_rows_C * gpuCols * sizeof(float),cudaMemAttachGlobal);

    // Fill the matrices on CPU
    fill_2d_matrix(cpuMatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(cpuMatB, nr_rows_B, cpuCols,1.0);
    fill_2d_matrix(cpuMatC, nr_rows_C, cpuCols,0.0);

    // Fill the matrices on CPU
    fill_2d_matrix(gpuMatA, nr_rows_A, nr_cols_A,1.0);
    fill_2d_matrix(gpuMatB, nr_rows_B, gpuCols,1.0);
    fill_2d_matrix(gpuMatC, nr_rows_C, gpuCols,0.0);

    // Allocate counters
    long long real_time[2];
	float *run_real_time = (float*) malloc(sizeof(float)*repeats);
	float *run_proc_time = (float*) malloc(sizeof(float)*repeats);
    float allTime = 0.0;
	for (int i=0; i < repeats; i++){

		// Initialize time counters
		real_time[0] = PAPI_get_real_usec();

		// Launch the CPU and GPU threads
		int cpuThreadError = pthread_create(&(tid[0]), NULL, &cpuThreadWithMemSplit, NULL);
		int gpuThreadError = pthread_create(&(tid[1]), NULL, &gpuThreadWithMemSplit, NULL);
		pthread_join(tid[0], NULL);
		pthread_join(tid[1], NULL);
		cudaDeviceSynchronize();

		// Collect time counters
		real_time[1] = PAPI_get_real_usec();
		run_real_time[i] = convert_to_sec(real_time[1] - real_time[0]);
	}

	float averageTime = calculate_average(run_real_time, repeats);
 	if (showDetails){printf("FLOPS: %f GFLOPS\n", (2.0*dim*dim*dim/averageTime)*1e-9);
 	printf("Runtime: %f Seconds\n", averageTime);}

    if (showDetails == 1){
    	printf("Matrix C after multiplication\n");
		printf("C = \n");
		print_matrix(cpuMatC, nr_rows_C, cpuCols);
		print_matrix(gpuMatC, nr_rows_C, gpuCols);
    }

    // Free memory
    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);

    // Output the gflops and time passed
    printf("%f %f", (2.0*dim*dim*dim/averageTime)*1e-9, averageTime);

}

// Thread launching helpers
void* gpuThread(){

	// Multiply A and B on GPU
	gpu_blas_split(matA, matB, matC, nr_rows_A, nr_cols_A, nr_cols_B);
	return NULL;
}
void* gpuThreadWithMemSplit(){

	// Multiply A and B on GPU
	gpu_blas_mmul(gpuMatA, gpuMatB, gpuMatC, nr_rows_A, nr_cols_A, gpuCols);
	return NULL;
}
void* gpuThreadSyncMemCpy(){

	// Multiply A and B on GPU
	gpu_blas_split(mappedMatA, mappedMatB, mappedMatC, nr_rows_A, nr_cols_A, nr_cols_B);
	return NULL;
}
void* gpuThreadAsyncMemCpy(){

	// Multiply A and B on GPU
	gpu_blas_split_with_stream(mappedMatA, mappedMatB, mappedMatC, nr_rows_A, nr_cols_A, nr_cols_B);
	return NULL;
}
void* cpuThread(){

	// Multiply A and B on CPU
	cpu_blas_split(matA, matB, matC, nr_rows_A, nr_cols_A, nr_cols_B);
	return NULL;
}
void* cpuThreadWithMemSplit(){

	// Multiply A and B on CPU
	cpu_blas_mmul(cpuMatA, cpuMatB, cpuMatC, nr_rows_A, nr_cols_A, cpuCols);
	return NULL;
}
