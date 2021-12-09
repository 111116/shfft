#pragma once

#include <string>
#include "fourierseries.hpp"
#include "legendre.hpp"
#include "shorder.hpp"
#include "select_size.hpp"

const double PI = acos(-1);

// normalization constant K as in definition of SH basis
double K(int l, int m)
{
	return std::sqrt(double(2*l+1)*fac(l-m)/(4*PI*fac(l+m)));
}

FS cached_P_cosx[n][n];

void fs_P_init()
{
	for (int l=0; l<n; ++l)
		for (int m=0; m<=l; ++m)
			cached_P_cosx[l][m] = P_cosx(l,m);
}

FS fs_P(int l, int m) // fs_P_init must be called first
{
	return cached_P_cosx[l][m];
}

static int SHIndex(int l, int m) {
	return l*l+l+m;
}

static int FS5_Index(int a, int b, int order)
{
	// DFT size: N*N
	constexpr int N = select_size_5(n);
	int offset = order - 1;
	return (a+offset) * N + (b+offset);
}