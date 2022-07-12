#define FMA1 s = n+m*s;
#define FMA2 FMA1\
    FMA1
#define FMA4 FMA2\
    FMA2
#define FMA8 FMA4\
    FMA4
#define FMA16 FMA8\
    FMA8
#define FMA32 FMA16\
    FMA16
#define FMA64 FMA32\
    FMA32
__global__ void MAXFLOPS_andrew(float *a, int len, int rotations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float s;
    float m;
    float n;
    if(idx+2<len) {
        s=a[idx];
        m=a[idx+1];
        n=a[idx+2];
    }
    for(int i=0; i<rotations; i++) {
        FMA64
    }
    a[idx] = s;
}
