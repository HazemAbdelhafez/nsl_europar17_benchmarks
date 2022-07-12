/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)                 
    SCALE:      a(i) = q*b(i)               
    SUM:        a(i) = b(i) + c(i)          
    TRIAD:      a(i) = b(i) + q*c(i)        

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation
  Modified by: Douglas Enright (dpephd-nvidia@yahoo.com), 1 December 2010
  Extensive Revisions, 4 December 2010

  User interface motivated by bandwidthTest NVIDIA SDK example.
*/

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <cutil_inline.h>
#include <shrUtils.h>

#define N	2000000
#define NTIMES	10

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

const float flt_eps = 1.192092896e-07f;

__global__ void set_array(float *a,  float value, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy(float *a, float *b, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = a[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy_Optimized(float *a, float *b, size_t len)
{
  /* 
   * Ensure size of thread index space is as large as or greater than 
   * vector index space 
   */
  if (blockDim.x * gridDim.x > len) return; 
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = a[idx];
}

__global__ void STREAM_Scale(float *a, float *b, float scale,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = scale* a[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Add( float *a, float *b, float *c,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Triad( float *a, float *b, float *c, float scalar, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+scalar*b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

/* Host side verification routines */
bool STREAM_Copy_verify(float *a, float *b, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult = a[idx];
    float diffResultExpected = (b[idx] - expectedResult);
    float relErrorULPS = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }  

  return bDifferent;
}

bool STREAM_Scale_verify(float *a, float *b, float scale, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult = scale*a[idx];
    float diffResultExpected = (b[idx] - expectedResult);
    float relErrorULPS = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }  

  return bDifferent;
}

bool STREAM_Add_verify(float *a, float *b, float *c, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult = a[idx] + b[idx];
    float diffResultExpected = (c[idx] - expectedResult);
    float relErrorULPS = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }

  return bDifferent;
}

bool STREAM_Triad_verify(float *a, float *b, float *c, float scalar, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult = a[idx] + scalar*b[idx];
    float diffResultExpected = (c[idx] - expectedResult);
    float relErrorULPS = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 3.f);
  }

  return bDifferent;
}

/* forward declarations */
int setupStream(const int argc, const char **argv);
void runStream(bool bDontUseGPUTiming);
void printResultsReadable(float times[][NTIMES]);
void printHelp(void);

int main(int argc, char *argv[])
{
 
  printf("[Single-Precision Device-Only STREAM Benchmark implementation in CUDA]\n");

  //set logfile name and start logs
  shrSetLogFileName("streamBenchmark.txt");
  shrLog("%s Starting...\n\n", argv[0]);

  int iRetVal = setupStream(argc, (const char**)argv);
  if (iRetVal != -1)
  {
    shrLog("\n[streamBenchmark] - results:\t%s\n\n", (iRetVal == 0) ? "PASSES" : "FAILED");
  }
  //finish
  shrEXIT(argc, (const char**)argv);
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int setupStream(const int argc, const char **argv)
{
  int deviceNum = 0;
  char *device = NULL;
  bool bDontUseGPUTiming = false;

  //process command line args
  if(shrCheckCmdLineFlag( argc, argv, "help"))
  {
    printHelp();
    return -1;
  }

  if( shrGetCmdLineArgumentstr(argc, argv, "device", &device) )
  {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if( deviceCount == 0 )
    {
      shrLog("!!!!!No devices found!!!!!\n");
      return -1000;
    } 
   
    deviceNum = atoi(device);
    if( deviceNum >= deviceCount || deviceNum < 0)
    {
      shrLog("\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", deviceNum, 0);
      deviceNum = 0;
    }
  }

  cudaSetDevice(deviceNum);
  shrLog("Running on...\n\n");
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, deviceNum) == cudaSuccess) 
    shrLog(" Device %d: %s\n", deviceNum, deviceProp.name);
    
  /*cutilSafeCall(cudaSetDeviceFlags(cudaDeviceBlockingSync));*/

  if(shrCheckCmdLineFlag( argc, argv, "cputiming")) {
    bDontUseGPUTiming = true;
    shrLog(" Using cpu-only timer.\n");
  }
	
  runStream(bDontUseGPUTiming);

  return 0;
}

///////////////////////////////////////////////////////////////////////////
// runStream
///////////////////////////////////////////////////////////////////////////
void runStream(bool bDontUseGPUTiming)
{
  float *d_a, *d_b, *d_c;

  int k;
  float times[5][NTIMES];
  float scalar;

  /* Allocate memory on device */
  cutilSafeCall( cudaMalloc((void**)&d_a, sizeof(float)*N) );
  cutilSafeCall( cudaMalloc((void**)&d_b, sizeof(float)*N) );
  cutilSafeCall( cudaMalloc((void**)&d_c, sizeof(float)*N) );

  /* Compute execution configuration */
  dim3 dimBlock(128); /* (128,1,1) */
  dim3 dimGrid(N/dimBlock.x); /* (N/dimBlock.x,1,1) */
  /* if( N % dimBlock.x != 0 ) dimGrid.x+=1; */

  shrLog(" Array size (single precision) = %u\n",N);
  shrLog(" using %u threads per block, %u blocks\n",dimBlock.x,dimGrid.x);

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
  unsigned int timer = 0;
  cudaEvent_t start, stop;

  /* both timers report msec */
  cutilCheckError( cutCreateTimer( &timer ) ); /* cpu (gettimeofday) timer */
  cutilSafeCall( cudaEventCreate( &start ) );  /* gpu timer facility */
  cutilSafeCall( cudaEventCreate( &stop ) );   /* gpu timer facility */

  scalar=3.0f;
  for (k=0; k<NTIMES; k++)
  {

    cutilCheckError( cutStartTimer( timer ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );
    STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
    cutilSafeCall( cudaEventRecord( stop, 0 ) );
    cutilSafeCall( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
    if (bDontUseGPUTiming) {
      times[0][k] = cutGetTimerValue( timer );
    } else {
      cutilSafeCall( cudaEventElapsedTime( &times[0][k], start, stop ) );
    }
 
    cutilCheckError( cutStartTimer( timer ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );
    STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
    cutilSafeCall( cudaEventRecord( stop, 0 ) );
    cutilSafeCall( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
    if (bDontUseGPUTiming) {
      times[1][k] = cutGetTimerValue( timer );
    } else {
      cutilSafeCall( cudaEventElapsedTime( &times[1][k], start, stop ) );
    }
 
    cutilCheckError( cutStartTimer( timer ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );
    STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cutilSafeCall( cudaEventRecord( stop, 0 ) );
    cutilSafeCall( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
    cutilCheckError( cutStopTimer( timer ) ); 
    if (bDontUseGPUTiming) {
      times[2][k] = cutGetTimerValue( timer );
    } else {
      cutilSafeCall( cudaEventElapsedTime( &times[2][k], start, stop ) );
    }

    cutilCheckError( cutStartTimer( timer ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );
    STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cutilSafeCall( cudaEventRecord( stop, 0 ) );
    cutilSafeCall( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
    cutilCheckError( cutStopTimer( timer ) );
    if (bDontUseGPUTiming) {
      times[3][k] = cutGetTimerValue( timer );
    } else {
      cutilSafeCall( cudaEventElapsedTime( &times[3][k], start, stop ) );
    }

    cutilCheckError( cutStartTimer( timer ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );
    STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cutilSafeCall( cudaEventRecord( stop, 0 ) );
    cutilSafeCall( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
    cutilCheckError( cutStopTimer( timer ) );
    if (bDontUseGPUTiming) {
      times[4][k] = cutGetTimerValue( timer );
    } else {
      cutilSafeCall( cudaEventElapsedTime( &times[4][k], start, stop ) );
    }	

  }

  /* verify kernels */
  float *h_a, *h_b, *h_c;
  bool errorSTREAMkernel = true;

  if ( (h_a = (float*)calloc( N, sizeof(float) )) == (float*)NULL ) {
    printf("Unable to allocate array h_a, exiting ...\n");
    exit(1);
  }
  if ( (h_b = (float*)calloc( N, sizeof(float) )) == (float*)NULL ) {
    printf("Unable to allocate array h_b, exiting ...\n");
    exit(1);
  }

  if ( (h_c = (float*)calloc( N, sizeof(float) )) == (float*)NULL ) {
    printf("Unalbe to allocate array h_c, exiting ...\n");
    exit(1);
  }

  /* 
   * perform kernel, copy device memory into host memory and verify each 
   * device kernel output 
   */
  STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
  cutilSafeCall( cudaMemcpy( h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    shrLog(" device STREAM_Copy:\t\tError detected in device STREAM_Copy, exiting\n");
    exit(-2000);
  } else {
    shrLog(" device STREAM_Copy:\t\tPass\n"); 
  }
  
  STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
  cutilSafeCall( cudaMemcpy( h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    shrLog(" device STREAM_Copy_Optimized:\tError detected in device STREAM_Copy, exiting\n");
    exit(-2000);
  } else {
    shrLog(" device STREAM_Copy_Optimized:\tPass\n"); 
  }
  
  STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar, N); 
  cutilSafeCall( cudaMemcpy( h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Scale_verify(h_b, h_c, scalar, N);
  if (errorSTREAMkernel) {
    shrLog(" device STREAM_Scale:\t\tError detected in device STREAM_Scale, exiting\n");
    exit(-3000);
  } else {
    shrLog(" device STREAM_Scale:\t\tPass\n");
  }

  STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N); 
  cutilSafeCall( cudaMemcpy( h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Add_verify(h_a, h_b, h_c, N);
  if (errorSTREAMkernel) {
    shrLog(" device STREAM_Add:\t\tError detected in device STREAM_Add, exiting\n");
    exit(-4000);
  } else {
    shrLog(" device STREAM_Add:\t\tPass\n");
  }

  STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar, N); 
  cutilSafeCall( cudaMemcpy( h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  cutilSafeCall( cudaMemcpy( h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Triad_verify(h_b, h_c, h_a, scalar, N);
  if (errorSTREAMkernel) {
    shrLog(" device STREAM_Triad:\t\tError detected in device STREAM_Triad, exiting\n");
    exit(-5000);
  } else {
    shrLog(" device STREAM_Triad:\t\tPass\n");
  }

  /* continue from here */
  printResultsReadable(times);
 
  //clean up timers
  cutilCheckError( cutDeleteTimer( timer ) );
  cutilSafeCall(cudaEventDestroy( stop ) );
  cutilSafeCall(cudaEventDestroy( start ) );
 
  /* Free memory on device */
  cutilSafeCall( cudaFree(d_a) );
  cutilSafeCall( cudaFree(d_b) );
  cutilSafeCall( cudaFree(d_c) );

}

///////////////////////////////////////////////////////////////////////////
//Print Results to Screen and File
///////////////////////////////////////////////////////////////////////////
void printResultsReadable(float times[][NTIMES]) {

  int j,k;

  float	avgtime[5] = {0., 0., 0., 0., 0.}, maxtime[5] = {0., 0., 0., 0., 0.},
	mintime[5] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


  char	*label[5] = {"Copy:      ", "Copy Opt.: ", "Scale:     ", "Add:       ", "Triad:     "};

  float	bytes_per_kernel[5] = { 
    2. * sizeof(float) * N,
    2. * sizeof(float) * N,
    2. * sizeof(float) * N,
    3. * sizeof(float) * N,
    3. * sizeof(float) * N
  }; 

  /* --- SUMMARY --- */

  for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
  {
    for (j=0; j<5; j++)
    {
      avgtime[j] = avgtime[j] + (1.e-03f * times[j][k]);
      mintime[j] = MIN(mintime[j], (1.e-03f * times[j][k]));
      maxtime[j] = MAX(maxtime[j], (1.e-03f * times[j][k]));
    }
  }
 
  shrLog("Function    Rate (GB/s)    Avg time      Min time      Max time\n");
  
  for (j=0; j<5; j++) {
     avgtime[j] = avgtime[j]/(float)(NTIMES-1);
     
     shrLog("%s%12.6e  %12.6e  %12.6e  %12.6e\n", label[j], 1.0E-09 * bytes_per_kernel[j]/mintime[j], avgtime[j], mintime[j], maxtime[j]);
  }
  
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
  shrLog("Usage:  streamsp [OPTION]...\n");
  shrLog("STREAM Benchmark implementation in CUDA\n");
  shrLog("Performs Copy, Scale, Add, and Triad single-precision kernels\n");
  shrLog("\n");
  shrLog("Example: ./streamsp\n");
  shrLog("\n");
  shrLog("Options:\n");
  shrLog("--help\t\t\tDisplay this help menu\n");
  shrLog("--device=[deviceno]\tSpecify the device to be used (Default: device 0)\n");
  shrLog("--cputiming\t\tForce CPU-based timing to be used\n");
}
