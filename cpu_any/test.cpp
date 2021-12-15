// This program is a proof-of-concept demonstration of computing SH products
// with arbitary k value, using the divide-and-conquer strategy. To obtain
// better performance, order of computation may be arranged with care,
// especially for small fixed k value.

#include <iostream>
#include <cmath>
#include <vector>
#include "lib/consolelog.hpp"
#include "fourierseries.hpp"
#include "fastmul.hpp"
#include "sh.hpp"
#include "precompute_sh2fs.hpp"
#include "precompute_fs2sh.hpp"
#include "readgamma.hpp"


typedef FourierSeries<float> FS;

FS sh2fs(const SH& sh)
{
    int n = sh.n;
    FS fs = FS::zeros(sh.n);
    for (int l=0; l<n; ++l)
    for (int m=-l; m<=l; ++m) {
        int i = SHIndex(l,m);
        for (int a=-n+1; a<n; ++a) {
            fs.at(a,-m) += sh.a[i] * coeff_sh2fs[i*4*n + a+n-1];
            fs.at(a, m) += sh.a[i] * coeff_sh2fs[i*4*n + a+n-1 + 2*n];
        }
    }
    return fs;
}

SH fs2sh(int n, int k, const FS& fs)
{
    SH sh(n);
    for (int l=0; l<n; ++l)
    for (int m=-l; m<=l; ++m) {
        int i = SHIndex(l,m);
        int b = -std::abs(m);
        for (int a=-(k*n-k); a<=0; ++a) {
            auto t = coeff_fs2sh[i * k*n -a];
            auto x = fs.at(a,b);
            sh.a[i] += t.real() * x.real() - t.imag() * x.imag();
        }
    }
    return sh;
}

using std::vector;

SH reference_prod(vector<SH> inputs)
{
    // checks
    int n_elem = inputs.size();
    if (n_elem == 0) throw "bad k";
    int n = inputs[0].n;
    if (n==0) throw "bad n";
    for (auto s: inputs)
        if (s.n != n)
            throw "input of varying order unsupported";
    if (sizeof(inputs[0].a.back()) != sizeof(float))
        throw "bad data type";
    // compute reference
    int N = (n-1)*((n_elem+1)/2)+1;
    auto gamma = readGamma(N);
    float A[N*N], B[N*N], C[N*N];
    memset(A, 0, N*N*sizeof(float));
    memcpy(A, inputs[0].a.data(), n*n*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    for (int i=1; i<inputs.size(); ++i) {
        memcpy(B, inputs[i].a.data(), n*n*sizeof(float));
        memset(C, 0, N*N*sizeof(float));
        for (auto e: gamma)
            C[e.c] += e.val * A[e.a] * B[e.b];
        memcpy(A, C, N*N*sizeof(float));
    }
    SH ref(n);
    memcpy(ref.a.data(), A, n*n*sizeof(float));
    return ref;
}

int main()
{
    // n: order of SH (i.e. n^2 coefficients each SH)
    // Result is (k-1) SH functions multiplied, reprojected onto n-th SH basis.
    int n=4, k=10;
    // Note: n and k can be assigned arbitarily at runtime.
    //       Here we demonstrate runtime precomputation.
    //       Performance of this demo may be suboptimal.
    console.log("n=", n);
    console.log("k=", k);
    precompute_sh2fs(n);
    // Note: if k is too large (i.e. product of many terms is required),
    //       it's usually reasonable to truncate intermediate results
    //       to avoid precomputing a very large conversion for the final result.
    precompute_fs2sh(n,k);

    // generate random multiplicands
    vector<SH> inputs;
    for (int i=0; i<k-1; ++i) {
        SH sh(n);
        for (int l=0; l<n; ++l)
            for (int m=-l; m<=l; m++) {
                sh.at(l,m) = (float)rand()/RAND_MAX-0.5;
            }
        inputs.push_back(sh);
    }

    // run our algorithm
    // step 1. convert to Fourier series
    vector<FS> fs;
    for (auto s: inputs)
        fs.push_back(sh2fs(s));
    // step 2. compute product of Fourier series
    FS prod = prod_divide_and_conquer(fs);
    // step 3. convert back to SH
    SH res = fs2sh(n,k,prod);

    // check correctness
    SH ref = reference_prod(inputs);
    console.log("err:", (ref-res).magnitude() / ref.magnitude());
}
