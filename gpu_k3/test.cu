#include <vector>
#include <cuda.h>
#include <curand.h>
#include <assert.h>
#include "lib/consolelog.hpp"
#include "shorder.hpp"
#include "shproduct.hpp"
#include "readgamma.hpp"
#include "shfft.cu"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// CUDA kernel, performs conventional O(n^5) multiplicatoin of SH vectors
// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...

__constant__ TensorEntry* deviceSparseGamma;
__constant__ int deviceSparseGammaSize;
std::vector<TensorEntry> SparseGamma;


void initGamma()
{
    std::vector<TensorEntry> v,v1;
    int size = 0;
    TensorEntry* p;
    // load sparse gamma n
    v = readGamma(n);
    size = v.size();
    console.log("sparse n size:", size);
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGammaSize, &size, sizeof(int));
    gpuErrchk( cudaPeekAtLastError() );
    SparseGamma = v;
}

void releaseGamma()
{
    TensorEntry* p;
    cudaMemcpyFromSymbol(&p, deviceSparseGamma, sizeof(TensorEntry*));
    cudaFree(p);
}

__global__ void shprod_conventional(float* A, float* B, float* C)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n*n];
    float Breg[n*n];
    float Creg[n*n];
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    memset(Creg, 0, n*n*sizeof(float));
#define e deviceSparseGamma[i]
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Creg[e.c] += e.val * Areg[e.a] * Breg[e.b];
#undef e
    memcpy(C+base, Creg, n*n*sizeof(float));
}



float compare(float* A, float* B)
{
    float sum = 0, sumdiff = 0;
    for (int i=0; i<n*n; ++i) {
        sumdiff += (A[i] - B[i]) * (A[i] - B[i]);
        sum += A[i] * A[i];
    }
    sum = std::sqrt(sum);
    sumdiff = std::sqrt(sumdiff);
    return sumdiff / sum;
}

void validate_sh2fs(float* deviceA)
{
    cufftComplex *pool0;
    cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*(num+1));
    float* B;
    cudaMalloc((void**)&B, sizeof(float)*n*n*num);
    // call
    const int blocksize = 32;
    const int extrasize = ((2*n-2)*N+(2*n-2)) - ((n-1)*N+(n-1));
    cu_sh2fs<<<num/blocksize, blocksize>>>(deviceA, pool0+extrasize);
    cudaMemset(pool0, 0, extrasize * sizeof(cufftComplex));
    cudaDeviceSynchronize();
    cu_fs2sh<<<num/blocksize, blocksize>>>(pool0, B);
    cudaDeviceSynchronize();
    // validate
    float* p = new float[n*n];
    float* p1 = new float[n*n];
    cudaMemcpy(p, deviceA, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0; i<n*n; ++i)
    //     std::cout << p[i] << " ";
    // puts("");
    cudaMemcpy(p1, B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        p1[i] *= N;
    // for (int i=0; i<n*n; ++i)
    //     std::cout << p1[i] << " ";
    // puts("");
    console.warn("conversion Er:", compare(p,p1));
}



// return relative error of kth result
float validate_err(float* deviceA, float* deviceB, float* deviceC, int k)
{

    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];
    gpuErrchk(cudaMemcpy(A, deviceA + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(B, deviceB + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(C, deviceC + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    float* ref = new float[n*n];
    // compute reference on CPU
    memset(ref, 0, n*n*sizeof(float));
    for (auto e: SparseGamma)
        ref[e.c] += e.val * A[e.a] * B[e.b];
    // print debug
    auto printarr = [&](float* A) {
        for (int i=0; i<n*n; ++i)
            std::cout << A[i] << " ";
        std::cout << "\n";
    };
    // printarr(A);
    // printarr(B);
    // printarr(C);
    // printarr(ref);
    // compute difference
    return compare(ref, C);
}

// validate result C = A * B
void validate(float* deviceA, float* deviceB, float* deviceC)
{
    float sum = 0;
    const int n_sample = 100;
    for (int i=0; i<n_sample; ++i)
        sum += validate_err(deviceA, deviceB, deviceC, rand()%num);
    float err_avg = sum / n_sample;
    console.log("err:", err_avg);
}

int main(int argc, char* argv[])
{
    console.log("order:",n);
    initGamma();
    // alloc
    float *A, *B, *C;
    cudaMalloc(&A, num * n*n * sizeof(float));
    cudaMalloc(&B, num * n*n * sizeof(float));
    cudaMalloc(&C, num * n*n * sizeof(float));
    // random init
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    curandGenerateUniform(curand, A, num * n*n);
    curandGenerateUniform(curand, B, num * n*n);

    validate_sh2fs(A);

    // set threading dimensions
    const int blocksize = 32;
    assert(num%blocksize==0);
    dim3 grid(num/blocksize,1);
    dim3 block(blocksize,1);
    // run traditional
    console.info("trad:");
    console.time("run " + std::to_string(num));
    shprod_conventional<<<grid, block>>>(A,B,C);
    cudaDeviceSynchronize();
    console.timeEnd("run " + std::to_string(num));
    gpuErrchk( cudaPeekAtLastError() );
    // validate
    validate(A,B,C);
    // clear result
    curandGenerateUniform(curand, C, num * n*n);
    releaseGamma();
    // run ours
    console.info("ours:");
    console.time("include fft planning " + std::to_string(num));
    shprod_many(A,B,C);
    console.timeEnd("include fft planning " + std::to_string(num));
    gpuErrchk( cudaPeekAtLastError() );
    // validate
    validate(A,B,C);
}
