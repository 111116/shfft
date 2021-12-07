#include <cufft.h>
#include <chrono> // time
#include "shorder.hpp"
#include "select_size.hpp"

// DFT size: N*N
constexpr int N = select_size_3(n);

// convert from n*n SH vector to coefficients of Fourier Series
// placed at lower-most corner in the N*N array
__global__ void cu_sh1_fs3(float* SH, cufftComplex* FS)
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
    #include "generated/sh1_fs3.cu"
    // copy back to global memory
   	for (int j=0; j<N*N; ++j)
   		FS[j+i*N*N] = FSreg[j];
}

// convert from coefficients of Fourier Series to SH vector
__global__ void cu_fs3_sh1(cufftComplex* FS, float* SH)
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
    #include "generated/fs3_sh1.cu"
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
void shprod_many(float* A, float* B, float* C, float* D)
{
	auto t0 = std::chrono::system_clock::now();
	auto t1 = std::chrono::system_clock::now();
	double dt = 0;
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
    console.time("exclude_planning " + std::to_string(num));
	// DFT on A
	cu_sh1_fs3<<<num/blocksize, blocksize>>>(A, pool0);
#define STARTTIME {cudaDeviceSynchronize(); t0 = std::chrono::system_clock::now();}
#define ENDTIME {cudaDeviceSynchronize(); t1 = std::chrono::system_clock::now(); dt += std::chrono::duration<double>(t1-t0).count()*1000;}
STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
ENDTIME
	// DFT on B
	cu_sh1_fs3<<<num/blocksize, blocksize>>>(B, pool0);
STARTTIME
	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	// element-wise multiply
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
ENDTIME
	// DFT on C
	cu_sh1_fs3<<<num/blocksize, blocksize>>>(C, pool0);
STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	// element-wise multiply
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);
ENDTIME
	cu_fs3_sh1<<<num/blocksize, blocksize>>>(pool1, D);
	// synchronize
	cudaDeviceSynchronize();
	console.log("fftexec:", dt);
    console.timeEnd("exclude_planning " + std::to_string(num));
}
