#include <iostream>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void checkGMem() // print available video memory
{
	size_t free, total;
	gpuErrchk(cudaMemGetInfo(&free, &total));
	std::cerr << "available memory: " << free << " / " << total << "\n";
}

const int N = 16384;

__global__ void test_kernel(float* A)
{
    // create thread local memory
    float reg[N];
    memset(reg, 0, N*sizeof(float));
    // copy back to global memory
   	for (int j=0; j<N; ++j)
   		A[j] = reg[j];
}

int main()
{
	checkGMem();
	// alloc
	float *pool0;
	cudaMalloc((void**)&pool0, sizeof(float)*N);

	checkGMem(); // 剩余显存还有11G
	test_kernel<<<1,1>>>(pool0);
	checkGMem(); // 剩余显存只剩6.6G
	cudaDeviceSynchronize();
	checkGMem(); // 剩余显存只剩6.6G
    gpuErrchk( cudaPeekAtLastError() );
}
