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
__constant__ TensorEntry* deviceSparseGamma213;
__constant__ int deviceSparseGamma213Size;
__constant__ TensorEntry* deviceSparseGamma314;
__constant__ int deviceSparseGamma314Size;
__constant__ TensorEntry* deviceSparseGamma411;
__constant__ int deviceSparseGamma411Size;
// on CPU
std::vector<TensorEntry> SparseGamma3;

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
    v = readGamma(3*n-2);
    size = v.size();
    console.log("sparse 3n-2 size:", size);
    // gamma 1,1,2
    v1 = filterGamma(v, n, n, 2*n-1);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma112, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma112Size, &size, sizeof(int));
    // gamma 2,1,3
    v1 = filterGamma(v, 2*n-1, n, 3*n-2);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma213, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma213Size, &size, sizeof(int));
    // set gamma on CPU
    gpuErrchk( cudaPeekAtLastError() );
    SparseGamma3 = v;
    //
    v = readGamma(4*n-3);
    // gamma 3,1,4
    v1 = filterGamma(v, 3*n-2, n, 4*n-3);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma314, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma314Size, &size, sizeof(int));
    // gamma 4,1,1
    v1 = filterGamma(v, 4*n-3, n, n);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma411, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma411Size, &size, sizeof(int));
}

void releaseGamma()
{
    TensorEntry* p;
    cudaMemcpyFromSymbol(&p, deviceSparseGamma, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma112, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma213, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma314, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma411, sizeof(TensorEntry*));
    cudaFree(p);
}

__global__ void shprod_conventional(float* A, float* B, float* C, float* D, float* E, float* F)
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
    memcpy(Areg, D+base, n*n*sizeof(float));
    memset(Treg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Treg[e.c] += e.val * Areg[e.a] * Breg[e.b];
    memcpy(Areg, E+base, n*n*sizeof(float));
    memset(Breg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Breg[e.c] += e.val * Treg[e.a] * Areg[e.b];
#undef e
    memcpy(F+base, Breg, n*n*sizeof(float));
}


__global__ void shprod_conventional_precise(float* A, float* B, float* C, float* D, float* E, float* F)
{
    // bruteforce, not yet optimized
    constexpr int n1 = 4*n-3;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n1*n1];
    float Breg[n1*n1];
    float T1reg[n1*n1];
    float T2reg[n1*n1];
    memset(Areg, 0, n1*n1*sizeof(float));
    memset(Breg, 0, n1*n1*sizeof(float));
#define SHMUL(A,B,C,ka,kb,kc) \
    memset(C, 0, n1*n1*sizeof(float)); \
    for (int i=0; i<deviceSparseGamma##ka##kb##kc##Size; ++i) \
        C[deviceSparseGamma##ka##kb##kc[i].c] += deviceSparseGamma##ka##kb##kc[i].val * A[deviceSparseGamma##ka##kb##kc[i].a] * B[deviceSparseGamma##ka##kb##kc[i].b];
    // T2 = A * B
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    SHMUL(Areg, Breg, T2reg, 1,1,2);
    // T1 = T2 * C
    memcpy(Areg, C+base, n*n*sizeof(float));
    SHMUL(T2reg, Areg, T1reg, 2,1,3);
    // T2 = T1 * D
    memcpy(Areg, D+base, n*n*sizeof(float));
    SHMUL(T1reg, Areg, T2reg, 3,1,4);
    // Areg = T2 * E
    memcpy(Breg, E+base, n*n*sizeof(float));
    SHMUL(T2reg, Breg, Areg, 4,1,1);
    memcpy(F+base, Areg, n*n*sizeof(float));
}


void validate_sh1_fs5(float* deviceA)
{
    cufftComplex *pool0;
    cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*(num+1));
    float* B;
    cudaMalloc((void**)&B, sizeof(float)*n*n*num);
    // call
    const int blocksize = 32;
    const int extrasize = ((4*n-4)*N+(4*n-4)) - ((n-1)*N+(n-1));
    cu_sh1_fs5<<<num/blocksize, blocksize>>>(deviceA, pool0+extrasize);
    cudaMemset(pool0, 0, extrasize * sizeof(cufftComplex));
    gpuErrchk(cudaDeviceSynchronize());
    cu_fs5_sh1<<<num/blocksize, blocksize>>>(pool0, B);
    gpuErrchk(cudaDeviceSynchronize());
    // validate
    float* p = new float[n*n];
    cudaMemcpy(p, deviceA, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i] << " ";
    puts("");
    cudaMemcpy(p, B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i]*N*N << " ";
    puts("");
}


void compute_reference(float* A, float* B, float* C, float* D, float* E, float* res)
{
    // SH<3*n-2> A1,B1,C1,D1,E1;
    // memcpy(A1.a, A, n*n*sizeof(float));
    // memcpy(B1.a, B, n*n*sizeof(float));
    // memcpy(C1.a, C, n*n*sizeof(float));
    // memcpy(D1.a, D, n*n*sizeof(float));
    // memcpy(E1.a, E, n*n*sizeof(float));
    // SH<3*n-2> ref = A1*B1*C1*D1*E1;
    // memcpy(res, ref.a, n*n*sizeof(float));
    // return;

    // compute reference on CPU, un-optimized
    // convert them all to (3n-2) order
    const int n1 = 3*n-2; /////// TODO
    float A1[n1*n1], B1[n1*n1], C1[n1*n1], D1[n1*n1], E1[n1*n1];
    float M1[n1*n1], M2[n1*n1], M3[n1*n1];
    memset(A1, 0, n1*n1*sizeof(float));
    memset(B1, 0, n1*n1*sizeof(float));
    memset(C1, 0, n1*n1*sizeof(float));
    memset(D1, 0, n1*n1*sizeof(float));
    memset(E1, 0, n1*n1*sizeof(float));
    memset(M1, 0, n1*n1*sizeof(float));
    memset(M2, 0, n1*n1*sizeof(float));
    memset(M3, 0, n1*n1*sizeof(float));
    memcpy(A1, A, n*n*sizeof(float));
    memcpy(B1, B, n*n*sizeof(float));
    memcpy(C1, C, n*n*sizeof(float));
    memcpy(D1, D, n*n*sizeof(float));
    memcpy(E1, E, n*n*sizeof(float));
    // M2 = A1 * B1
    for (auto e: SparseGamma3)
        M2[e.c] += e.val * A1[e.a] * B1[e.b];
    // M1 = M1 * C1
    for (auto e: SparseGamma3)
        M1[e.c] += e.val * M2[e.a] * C1[e.b];
    // M2 = D1 * E1
    memset(M2, 0, n1*n1*sizeof(float));
    for (auto e: SparseGamma3)
        M2[e.c] += e.val * D1[e.a] * E1[e.b];
    // M3 = M1 * M2 (order matters!)
    for (auto e: SparseGamma3)
        M3[e.c] += e.val * M1[e.a] * M2[e.b];
    // copy result
    memcpy(res, M3, n*n*sizeof(float));
}

// return relative error of kth result
float validate_err(float* deviceA, float* deviceB, float* deviceC, float* deviceD, float* deviceE, float* deviceF, int k)
{
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];
    float* D = new float[n*n];
    float* E = new float[n*n];
    float* F = new float[n*n];
    gpuErrchk(cudaMemcpy(A, deviceA + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(B, deviceB + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(C, deviceC + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(D, deviceD + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(E, deviceE + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(F, deviceF + k*n*n, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    float* ref = new float[n*n];
    memset(ref, 0, n*n*sizeof(float));
    compute_reference(A, B, C, D, E, ref);
    // print debug
    auto printarr = [&](float* A) {
        for (int i=0; i<n*n; ++i)
            std::cout << A[i] << " ";
        std::cout << "\n";
    };
    // printarr(ref);
    // printarr(F);
    // compute difference
    float sum = 0, sumdiff = 0;
    for (int i=0; i<n*n; ++i) {
        sumdiff += (ref[i] - F[i]) * (ref[i] - F[i]);
        sum += ref[i] * ref[i];
    }
    sum = std::sqrt(sum);
    sumdiff = std::sqrt(sumdiff);
    return sumdiff / sum;
}

// validate result C = A * B
void validate(float* deviceA, float* deviceB, float* deviceC, float* deviceD, float* deviceE, float* deviceF)
{
    float sum = 0;
    const int n_sample = 10;
    for (int i=0; i<n_sample; ++i)
        sum += validate_err(deviceA, deviceB, deviceC, deviceD, deviceE, deviceF, rand()%num);
    float err_avg = sum / n_sample;
    console.log("err:", err_avg);
}



int main(int argc, char* argv[])
{
    console.log("order:",n);
    initGamma();
    // alloc
    float *A, *B, *C, *D, *E, *F;
    cudaMalloc(&A, num * n*n * sizeof(float));
    cudaMalloc(&B, num * n*n * sizeof(float));
    cudaMalloc(&C, num * n*n * sizeof(float));
    cudaMalloc(&D, num * n*n * sizeof(float));
    cudaMalloc(&E, num * n*n * sizeof(float));
    cudaMalloc(&F, num * n*n * sizeof(float));
    // random init
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    curandGenerateUniform(curand, A, num * n*n);
    curandGenerateUniform(curand, B, num * n*n);
    curandGenerateUniform(curand, C, num * n*n);
    curandGenerateUniform(curand, D, num * n*n);
    curandGenerateUniform(curand, E, num * n*n);

    // validate_sh1_fs5(A);

    // set threading dimensions
    const int blocksize = 32;
    assert(num%blocksize==0);
    dim3 grid(num/blocksize,1);
    dim3 block(blocksize,1);

    // run traditional
    console.time("trad");
    shprod_conventional<<<grid, block>>>(A,B,C,D,E,F);
    gpuErrchk(cudaDeviceSynchronize());
    console.timeEnd("trad");
    // validate
    validate(A,B,C,D,E,F);
    // clear result
    curandGenerateUniform(curand, F, num * n*n);
    releaseGamma();

    // run ours
    console.time("ours + planning");
    shprod_many(A,B,C,D,E,F);
    console.timeEnd("ours + planning");
    gpuErrchk( cudaPeekAtLastError() );
    // validate
    validate(A,B,C,D,E,F);
}
