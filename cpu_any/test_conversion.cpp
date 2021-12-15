#include "fourierseries.hpp"
#include "lib/consolelog.hpp"
#include "sh.hpp"

#include "precompute_sh2fs.hpp"
#include "precompute_fs2sh.hpp"

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

int main(int argc, char* argv[])
{
	int n = atoi(argv[1]);
    SH sh(n);
    for (int l=0; l<n; ++l)
        for (int m=-l; m<=l; m++) {
            sh.at(l,m) = (float)rand()/RAND_MAX-0.5;
        }
    console.time("p1");
    precompute_sh2fs(n);
    console.timeEnd("p1");
    console.time("p2");
    precompute_fs2sh(n,1);
    console.timeEnd("p2");
    FS fs = sh2fs(sh);
    SH sh1 = fs2sh(n,1,fs);
    console.log("err:", (sh1-sh).magnitude() / sh.magnitude());
}