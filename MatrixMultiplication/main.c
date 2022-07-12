#include "matmul.h"

void allTests(){

	// Start running the tests
	printf("Running matrix multiplication with unified memory on GPU\n");
	gpuMatMulWithUM();
	cudaDeviceReset();
	printf("Running matrix multiplication with explicit memory copy to the GPU\n");
	gpuMatMulWithMemCpy();
	cudaDeviceReset();
	printf("Running matrix multiplication with direct matrix allocation on the GPU memory\n");
	gpuMatMul();
	cudaDeviceReset();
	printf("Running matrix multiplication with unified memory on CPU\n");
	cpuMatMulWithUM();
	cudaDeviceReset();
	printf("Running matrix multiplication with explicit allocation on CPU memory\n");
	cpuMatMul();
	cudaDeviceReset();
	printf("Running matrix multiplication heterogeneously\n");
	heterogeneousWithHostAlloc();
}
void cpuTests(){

	// Start running the tests
	printf("Running matrix multiplication with explicit allocation on CPU memory\n");
	cpuMatMul();
	printf("Running matrix multiplication with unified memory on CPU\n");
	cpuMatMulWithUM();
	cudaDeviceReset();
	printf("Running matrix multiplication with CUDA HostAllocation on CPU\n");
	cpuMatMultWithHostAlloc();
	cudaDeviceReset();
}

void gpuTests(){

	// Start running the tests
	printf("Running matrix multiplication with direct matrix allocation on the GPU memory\n");
	gpuMatMul();
	cudaDeviceReset();
	printf("Running matrix multiplication with unified memory on GPU\n");
	gpuMatMulWithUM();
	cudaDeviceReset();
	printf("Running matrix multiplication with synch memory copy to the GPU\n");
	gpuMatMulWithMemCpy();
	cudaDeviceReset();
}

void heterogeneousTests(){

	// Start running the tests
	cudaDeviceReset();
	printf("Running matrix multiplication on both the CPU and GPU with HostMemAlloc\n");
	heterogeneousWithHostAlloc();
	cudaDeviceReset();
	printf("Running matrix multiplication on both the CPU and GPU with zero-copy memory\n");
	heterogeneousWithZeroMemCpy();
	cudaDeviceReset();
	printf("Running matrix multiplication on both the CPU and GPU with Asynch memory copy to the GPU\n");
	heterogeneousWithAsyncMemCpy();
	cudaDeviceReset();
	printf("Running matrix multiplication on both the CPU and GPU with Synch memory copy to the GPU\n");
	heterogeneousWithSyncMemCpy();
	cudaDeviceReset();
	printf("Running matrix multiplication on both the CPU and GPU with split matrix across different memories\n");
	heterogeneousWithSplitMem();
}

int main(int argc, char* argv[]) {

	showDetails = atoi(argv[1]);
	repeats = atoi(argv[2]);
	int benchmark = atoi(argv[3]);
	threads = atoi(argv[4]);
	dim = atoi(argv[5]);
	cpuCols = atoi(argv[6]);
	gpuCols = dim - cpuCols;
	// Initialize the GPU device and set it to 0
	cudaDeviceReset();
	cudaSetDevice(0);
	// Set number of threads for BLAS
	openblas_set_num_threads(threads);

   switch(benchmark) {
	  case 1 :
		if(showDetails)
		  printf("Running matrix multiplication with explicit allocation on CPU memory\n");
		cpuMatMul();
		break;
	  case 2 : // cancel from the test cases
		if(showDetails)
			printf("Running matrix multiplication with unified memory on CPU\n");
		cpuMatMulWithUM();
		cudaDeviceReset();
		break;
	  case 3 :	// cancel from the test cases
		if(showDetails)
			printf("Running matrix multiplication with CUDA HostAllocation on CPU\n");
		cpuMatMultWithHostAlloc();
		cudaDeviceReset();
		break;
	  case 4 :
		if(showDetails)
			printf("Running matrix multiplication with direct matrix allocation on the GPU memory\n");
		gpuMatMul();
		cudaDeviceReset();
		break;
	  case 5 :
		if(showDetails)
			printf("Running matrix multiplication with unified memory on GPU\n");
		gpuMatMulWithUM();
		cudaDeviceReset();
		break;
	  case 6 :
		if(showDetails)
			printf("Running matrix multiplication with synch memory copy to the GPU\n");
		gpuMatMulWithMemCpy();
		cudaDeviceReset();
		break;
	  case 7 :
		if(showDetails)
			printf("Running matrix multiplication on both the CPU and GPU with HostMemAlloc\n");
		heterogeneousWithHostAlloc();
		cudaDeviceReset();
		break;
	  case 8 :
		if(showDetails)
			printf("Running matrix multiplication on both the CPU and GPU with zero-copy memory\n");
		heterogeneousWithZeroMemCpy();
		cudaDeviceReset();
		break;
	  case 9 :
		if(showDetails)
			printf("Running matrix multiplication on both the CPU and GPU with Asynch memory copy to the GPU\n");
		heterogeneousWithAsyncMemCpy();
		cudaDeviceReset();
		break;
	  case 10 :
		if(showDetails)
			printf("Running matrix multiplication on both the CPU and GPU with Synch memory copy to the GPU\n");
		heterogeneousWithSyncMemCpy();
		cudaDeviceReset();
		break;
	  case 11 :
		if(showDetails)
			printf("Running matrix multiplication on both the CPU and GPU with split matrix across different memories\n");
		heterogeneousWithSplitMem();
		cudaDeviceReset();
		break;
	  default :
		 printf("Invalid Benchmark\n" );
   }
    return 0;
}
