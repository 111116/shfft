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
__constant__ TensorEntry* deviceSparseGamma112;
__constant__ int deviceSparseGamma112Size;
__constant__ TensorEntry* deviceSparseGamma211;
__constant__ int deviceSparseGamma211Size;
// on CPU
std::vector<TensorEntry> SparseGamma2;

std::vector<TensorEntry> filterGamma(std::vector<TensorEntry> v, int a, int b, int c)
{
    std::vector<TensorEntry> res;
    for (auto e: v) {
        if (e.a < a*a && e.b < b*b && e.c < c*c)
            res.push_back(e);
    }
    return res;
}

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
    // load sparse gamma 2n-1
    v = readGamma(2*n-1);
    size = v.size();
    console.log("sparse 2n-1 size:", size);
    // gamma 1,1,2
    v1 = filterGamma(v, n, n, 2*n-1);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma112, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma112Size, &size, sizeof(int));
    // gamma 2,1,1
    v1 = filterGamma(v, 2*n-1, n, n);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma211, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma211Size, &size, sizeof(int));
    // set gamma on CPU
    gpuErrchk( cudaPeekAtLastError() );
    SparseGamma2 = v;
}

__global__ void shprod_conventional(float* A, float* B, float* C, float* D)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n*n];
    float Breg[n*n];
    float Treg[n*n];
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    memset(Treg, 0, n*n*sizeof(float));
#define e deviceSparseGamma[i]
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Treg[e.c] += e.val * Areg[e.a] * Breg[e.b];
    memcpy(Areg, C+base, n*n*sizeof(float));
    memset(Breg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Breg[e.c] += e.val * Treg[e.a] * Areg[e.b];
#undef e
    memcpy(D+base, Breg, n*n*sizeof(float));
}


__global__ void shprod_conventional_precise(float* A, float* B, float* C, float* D)
{
    // bruteforce, not yet optimized
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    const int n1 = 2*n-1;
    float Areg[n1*n1];
    float Breg[n1*n1];
    float Treg[n1*n1];
    memset(Areg, 0, n1*n1*sizeof(float));
    memset(Breg, 0, n1*n1*sizeof(float));
    memset(Treg, 0, n1*n1*sizeof(float));
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
#define e deviceSparseGamma112[i]
    for (int i=0; i<deviceSparseGamma112Size; ++i)
        Treg[e.c] += e.val * Areg[e.a] * Breg[e.b];
#undef e
    memcpy(Areg, C+base, n*n*sizeof(float));
    memset(Breg, 0, n1*n1*sizeof(float));
#define e deviceSparseGamma211[i]
    for (int i=0; i<deviceSparseGamma211Size; ++i)
        Breg[e.c] += e.val * Treg[e.a] * Areg[e.b];
#undef e
    memcpy(D+base, Breg, n*n*sizeof(float));
}


void validate_sh1_fs3(float* deviceA)
{
    cufftComplex *pool0;
    cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*(num+1));
    float* B;
    cudaMalloc((void**)&B, sizeof(float)*n*n*num);
    // call
    const int blocksize = 32;
    const int extrasize = ((3*n-3)*N+(3*n-3)) - ((n-1)*N+(n-1));
    cu_sh1_fs3<<<num/blocksize, blocksize>>>(deviceA, pool0+extrasize);
    cudaMemset(pool0, 0, extrasize * sizeof(cufftComplex));
    cudaDeviceSynchronize();
    cu_fs3_sh1<<<num/blocksize, blocksize>>>(pool0, B);
    cudaDeviceSynchronize();
    // validate
    float* p = new float[n*n];
    cudaMemcpy(p, deviceA, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i] << " ";
    puts("");
    cudaMemcpy(p, B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i] << " ";
    puts("");
}


void compute_reference(float* A, float* B, float* C, float* res)
{
    // compute reference on CPU, un-optimized
    // convert them all to (3n-2) order
    const int n1 = 2*n-1;
    float A1[n1*n1], B1[n1*n1], C1[n1*n1];
    float M1[n1*n1], M2[n1*n1];
    memset(A1, 0, n1*n1*sizeof(float));
    memset(B1, 0, n1*n1*sizeof(float));
    memset(C1, 0, n1*n1*sizeof(float));
    memset(M1, 0, n1*n1*sizeof(float));
    memset(M2, 0, n1*n1*sizeof(float));
    memcpy(A1, A, n*n*sizeof(float));
    memcpy(B1, B, n*n*sizeof(float));
    memcpy(C1, C, n*n*sizeof(float));
    // M1 = A1 * B1
    for (auto e: SparseGamma2)
        M1[e.c] += e.val * A1[e.a] * B1[e.b];
    // M2 = M1 * C1
    for (auto e: SparseGamma2)
        M2[e.c] += e.val * M1[e.a] * C1[e.b];
    // copy result
    memcpy(res, M2, n*n*sizeof(float));
}

// return relative error of kth result
float validate_err(float* deviceA, float* deviceB, float* deviceC, float* deviceD, int k)
{
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];
    float* D = new float[n*n];
    gpuErrchk(cudaMemcpy(A, deviceA + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(B, deviceB + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(C, deviceC + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(D, deviceD + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    float* ref = new float[n*n];
    compute_reference(A, B, C, ref);
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
    // printarr(D);
    // compute difference
    float sum = 0, sumdiff = 0;
    for (int i=0; i<n*n; ++i) {
        // console.warn(C[i]);
        sumdiff += (ref[i] - D[i]) * (ref[i] - D[i]);
        sum += ref[i] * ref[i];
    }
    sum = std::sqrt(sum);
    sumdiff = std::sqrt(sumdiff);
    return sumdiff / sum;
}

// validate result C = A * B
void validate(float* deviceA, float* deviceB, float* deviceC, float* deviceD)
{
    float sum = 0;
    const int n_sample = 20;
    for (int i=0; i<n_sample; ++i)
        sum += validate_err(deviceA, deviceB, deviceC, deviceD, rand()%num);
    float err_avg = sum / n_sample;
    console.log("err:", err_avg);
}


void releaseGamma()
{
    TensorEntry* p;
    cudaMemcpyFromSymbol(&p, deviceSparseGamma, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma112, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma211, sizeof(TensorEntry*));
    cudaFree(p);
}

int main(int argc, char* argv[])
{
    console.log("order:",n);
    initGamma();
    // alloc
    float *A, *B, *C, *D;
    cudaMalloc(&A, num * n*n * sizeof(float));
    cudaMalloc(&B, num * n*n * sizeof(float));
    cudaMalloc(&C, num * n*n * sizeof(float));
    cudaMalloc(&D, num * n*n * sizeof(float));
    // random init
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    curandGenerateUniform(curand, A, num * n*n);
    curandGenerateUniform(curand, B, num * n*n);
    curandGenerateUniform(curand, C, num * n*n);

    // set threading dimensions
    const int blocksize = 32;
    assert(num%blocksize==0);
    dim3 grid(num/blocksize,1);
    dim3 block(blocksize,1);

    // run conventional
    console.info("conventional (approximate):");
    console.time("run " + std::to_string(num));
    shprod_conventional<<<grid, block>>>(A,B,C,D);
    cudaDeviceSynchronize();
    console.timeEnd("run " + std::to_string(num));
    // validate
    validate(A,B,C,D);
    // clear result
    curandGenerateUniform(curand, D, num * n*n);

    // run conventional precise
    console.info("conventional (precise):");
    console.time("run " + std::to_string(num));
    shprod_conventional_precise<<<grid, block>>>(A,B,C,D);
    cudaDeviceSynchronize();
    console.timeEnd("run " + std::to_string(num));
    // validate
    validate(A,B,C,D);
    // clear result
    curandGenerateUniform(curand, D, num * n*n);

    releaseGamma();
    // run ours
    console.info("ours:");
    console.time("run " + std::to_string(num));
    shprod_many(A,B,C,D);
    console.timeEnd("run " + std::to_string(num));
    gpuErrchk( cudaPeekAtLastError() );
    // validate
    validate(A,B,C,D);
}
