#include <cufft.h>
#include "shorder.hpp"
#include "select_size.hpp"

// DFT size: N*N
constexpr int N = select_size_5(n);

// convert from n*n SH vector to coefficients of Fourier Series
// placed at lower-most corner in the N*N array
__global__ void cu_sh1_fs5(float* SH, cufftComplex* FS)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
    cufftComplex FSreg[N*N];
    memcpy(SHreg, SH+shbase, n*n*sizeof(float));
    memset(FSreg, 0, N*N*sizeof(cufftComplex));
    // execute
    #include "generated/sh1_fs5.cu"
    // copy back to global memory
   	for (int j=0; j<N*N; ++j)
   		FS[j+i*N*N] = FSreg[j];
}

// convert from coefficients of Fourier Series to SH vector
__global__ void cu_fs5_sh1(cufftComplex* FS, float* SH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
    cufftComplex FSreg[N*N];
    memset(SHreg, 0, n*n*sizeof(float));
   	for (int j=0; j<N*N; ++j)
   		FSreg[j] = FS[j+i*N*N];
    // execute
    #include "generated/fs5_sh1.cu"
    // copy back to global memory
    memcpy(SH+shbase, SHreg, n*n*sizeof(float));
}

// element-wise multiplication B_i *= A_i
__global__ void multiply(cufftComplex* A, cufftComplex* B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = A[i].x * B[i].x - A[i].y * B[i].y;
	float y = A[i].y * B[i].x + A[i].x * B[i].y;
	B[i].x = x;
	B[i].y = y;
}

// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...
void shprod_many(float* A, float* B, float* C, float* D, float* E, float* F)
{
	auto t0 = std::chrono::system_clock::now();
	auto t1 = std::chrono::system_clock::now();
	const int blocksize = 32;
	assert(num%blocksize == 0);
	// mem alloc
	cufftComplex *pool0, *pool1, *pool2;
	cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*num);
	cudaMalloc((void**)&pool1, sizeof(cufftComplex)*N*N*num);
	cudaMalloc((void**)&pool2, sizeof(cufftComplex)*N*N*num);
	// plan DFT
	cufftHandle plan;
	int sizes[2] = {N,N};
	cufftPlanMany(&plan, 2, sizes, NULL, 1, N*N, NULL, 1, N*N, CUFFT_C2C, num);
    console.time("ours");
	// DFT on A
	cu_sh1_fs5<<<num/blocksize, blocksize>>>(A, pool0);
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	// DFT on B & multiply
	cu_sh1_fs5<<<num/blocksize, blocksize>>>(B, pool0);
	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// DFT on C & multiply
	cu_sh1_fs5<<<num/blocksize, blocksize>>>(C, pool0);
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// DFT on D & multiply
	cu_sh1_fs5<<<num/blocksize, blocksize>>>(D, pool0);
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// DFT on E & multiply
	cu_sh1_fs5<<<num/blocksize, blocksize>>>(E, pool0);
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);
	cu_fs5_sh1<<<num/blocksize, blocksize>>>(pool1, F);
	// synchronize
	cudaDeviceSynchronize();
    console.timeEnd("ours");
}
