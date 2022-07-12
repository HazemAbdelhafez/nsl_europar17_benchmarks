//Meant to be run with one thread and one block
__global__ void ANDREW_cache(float *a, int len, int rotations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float res=0;
    for(int j=0; j<rotations; j++){
        for(int i=0; i<len; i++) {
            res+=a[idx+i];
        }
        a[idx] = res;
    }
}
