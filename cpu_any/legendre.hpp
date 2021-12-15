// Associated_Legendre_polynomials
#pragma once

#include <unordered_map>
#include "fourierseries.hpp"


static int SHIndex(int l, int m) {
	return l*l+l+m;
}


namespace precompute {

typedef FourierSeries<double> FS;
using complex = FS::complex;

// factorial n!
double fac(int n)
{
	if (n<=1) return 1;
	return n*fac(n-1);
}

// generalized binomial coefficient
double Comb(double n, int k)
{
	double n_ = 1;
	for (int i=0; i<k; ++i)
		n_ *= n-i;
	return n_/fac(k);
}


// Fourier-series form of function x |-> P_l^m(cos(x))
// where P_l^m is associated Legendre polynomial
// Not optimized for performance
FS compute_P_cosx(int l, int m)
{
	if (not(l>=m && m>=0)) throw "err";
	static const FS fs_1(0,0,complex(1,0));
	static const FS fs_cos = FS(1,0,complex(0.5,0)) + FS(-1,0,complex(0.5,0));
	static const FS fs_sin = FS(1,0,complex(0,-0.5)) - FS(-1,0,complex(0,-0.5));
	// (1-cosx^2)^0.5 = sinx   for 0<x<pi
	FS c1 = std::pow(-1,m) * std::pow(2,l) * pow(fs_1 - pow(fs_cos, 2), m/2) * (m%2? fs_sin: fs_1);
	FS c2;
	for (int k=m; k<=l; ++k) {
		c2 += fac(k) / fac(k-m) * Comb(l,k) * Comb(0.5*(l+k-1), l) * pow(fs_cos, k-m);
	}
	return c1 * c2;
}

// cached version of compute_P_cosx
FS P_cosx(int l, int m)
{
	static std::unordered_map<int, FS> cache;
	int p = SHIndex(l,m);
	if (!cache.count(p))
		cache[p] = compute_P_cosx(l,m);
	return cache[p];
}


const double PI = acos(-1);

// normalization constant K as in definition of SH basis
double K(int l, int m)
{
	return std::sqrt(double(2*l+1)*fac(l-m)/(4*PI*fac(l+m)));
}


} // end namespace precompute

