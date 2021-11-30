// Associated_Legendre_polynomials
#pragma once

#include "fourierseries.hpp"
#include "shorder.hpp"

// factorial n!
float fac(int n)
{
	if (n<=1) return 1;
	return n*fac(n-1);
}

// generalized form of the binomial coefficient
float Comb(float n, int k)
{
	float n_ = 1;
	for (int i=0; i<k; ++i)
		n_ *= n-i;
	return n_/fac(k);
}

const int maxdeg = 3*n;
typedef FourierSeries<maxdeg> FS;

// Fourier-series form of function x |-> P_l^m(cos(x))
// where P_l^m is associated Legendre polynomial
// not optimized for performance
FS P_cosx(int l, int m)
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