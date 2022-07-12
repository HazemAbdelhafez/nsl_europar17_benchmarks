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
 */

#define N       (1024*1024*64)
#define N_ROTATION (1024*8)
#define NTIMES  4

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

__global__ void MAXFLOPS_andrew(float *a, int len, int rotations);
__global__ void ANDREW_cache(float *a, int len, int rotations);
static double	avgtime[5] = {0}, maxtime[5] = {0},
                mintime[5] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX, FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(float) * N,
    2 * sizeof(float) * N,
    3 * sizeof(float) * N,
    3 * sizeof(float) * N
};

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    double time = ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
    return time;
}


__global__ void set_array(float *a,  float value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) a[idx] = value;
}

__global__ void STREAM_Copy(float *a, float *b, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) b[idx] = a[idx];
}

__global__ void ANDREW_Copy(float *a, float *b, int len) {
    int stride = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    while(idx<len) {
        b[idx] = a[idx];
        idx+=stride;
    }
}
__global__ void STREAM_Scale(float *a, float *b, float scale,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) b[idx] = scale* a[idx];
}

__global__ void ANDREW_Scale(float *a, float *b, float scale,  int len)
{
    int stride = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx<len) {
        b[idx] = scale* a[idx];
        idx+=stride;
    }
}

__global__ void STREAM_Add( float *a, float *b, float *c,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) c[idx] = a[idx]+b[idx];
}

__global__ void ANDREW_Add( float *a, float *b, float *c,  int len)
{
    int stride = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx<len) {
        c[idx] = a[idx]+b[idx];
        idx+=stride;
    }
}

__global__ void STREAM_Triad( float *a, float *b, float *c, float scalar, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) c[idx] = a[idx]+scalar*b[idx];
}

__global__ void ANDREW_Triad( float *a, float *b, float *c, float scalar, int len)
{
    int stride = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx<len) {
        c[idx] = a[idx]+scalar*b[idx];
        idx+=stride;
    }
}
int readnflop(const char * fn) {
    FILE* file = fopen(fn, "r");
    int nmad=0, nadd=0;
    fscanf(file, "%d %d", &nmad, &nadd);
    fclose(file);
    return nmad*2+nadd;
}

int main()
{
    float *d_a, *d_b, *d_c;
    int j,k;
    double times[5][NTIMES];
    float scalar;

    printf(" STREAM Benchmark implementation in CUDA\n");
    printf(" Array size (single precision)=%d\n",N);

    /* Allocate memory on device */
    if(cudaSuccess!=cudaMalloc((void**)&d_a, sizeof(float)*N)) printf("ERROR\n");
    if(cudaSuccess!=cudaMalloc((void**)&d_b, sizeof(float)*N)) printf("ERROR\n");
    if(cudaSuccess!=cudaMalloc((void**)&d_c, sizeof(float)*N)) printf("ERROR\n");

    /* Compute execution configuration */
    dim3 dimBlock(128);
    dim3 dimGrid(N/dimBlock.x);
    if( N % dimBlock.x != 0 ) dimGrid.x+=1;

    printf(" using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    /* Initialize memory on the device */
    /*
    set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
    set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
    set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);
    */

    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
    dim3 andrew_block(1024);
    dim3 andrew_grid(4);
    scalar=3.0f;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k]= mysecond();
        ANDREW_Copy<<<andrew_grid, andrew_block>>>(d_a, d_c, N);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        times[0][k]= mysecond() -  times[0][k];

        times[1][k]= mysecond();
        ANDREW_Scale<<<andrew_grid,andrew_block>>>(d_b, d_c, scalar,  N);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        times[1][k]= mysecond() -  times[1][k];

        times[2][k]= mysecond();
        ANDREW_Add<<<andrew_grid,andrew_block>>>(d_a, d_b, d_c,  N);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        times[2][k]= mysecond() -  times[2][k];

        times[3][k]= mysecond();
        ANDREW_Triad<<<andrew_grid,andrew_block>>>(d_b, d_c, d_a, scalar,  N);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        times[3][k]= mysecond() -  times[3][k];

        times[4][k] = mysecond();
        MAXFLOPS_andrew<<<andrew_grid,andrew_block>>>(d_a, N, N_ROTATION);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        times[4][k]= mysecond() -  times[4][k];

    }

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
        for (j=0; j<5; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
                bytes[j]/(1000000*mintime[j]),
                avgtime[j],
                mintime[j],
                maxtime[j]);
    }
    int nflop = readnflop("maxflops.count");
    printf("Number of flops per iteration: %d\n", nflop);
    avgtime[4] = avgtime[4]/(double)(NTIMES-1);
    printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n",
            "MAX FLOPS: ",
            1.0*(nflop)*N_ROTATION*andrew_grid.x*andrew_block.x/(1000000*mintime[4]),
            avgtime[4],
            mintime[4],
            maxtime[4]);


    dim3 oneb(1);
    dim3 oneg(1);
    for(int len=1; (len*4)<=1*1024*1024; len*=2) {
        double time = mysecond();
        ANDREW_cache<<<oneg, oneb>>>(d_a, len, 128);
        if(cudaSuccess != cudaGetLastError()) printf("Error launching kernel\n");
        if(cudaDeviceSynchronize()!= cudaSuccess) printf("ERROR\n");
        time = mysecond()-time;
        printf("Size %d, mbytes/sec: %f\n", 4*len, (1.0*len*4*128)/(1000000*time));
    }
    /* Free memory on device */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
