// computes tripling coefficients of the real spherical harmonics described in pbrt-v2
#include "lib/wignerSymbols/include/wignerSymbols.h"
#include <complex>

using WignerSymbols::wigner3j;
using WignerSymbols::clebschGordan;

// https://github.com/code-google-com/cortex-vfx/blob/master/src/IECore/SphericalHarmonicsTensor.cpp

std::complex<double> U( int l, int m, int u )
{
	const static double invSqrt2 = 1/sqrt(2);
	double mS = (m&1?-invSqrt2:invSqrt2);
	if ( u == 0 )
	{
		if ( m == 0 )
			return 1;
	}
	else if ( u < 0 )
	{
		if ( m == u )
			return std::complex<double>( 0, mS );
		if ( m == -u )
			return std::complex<double>( 0, -invSqrt2 );
	}
	else if ( u > 0 )
	{
		if ( m == u )
			return invSqrt2;
		if ( m == -u )
			return mS;
	}
	return 0;
}

double gaunt( int Ji, int Mi, int Jj, int Mj, int Jk, int Mk )
{
	double A = sqrt( (double)(2*Ji+1)*(2*Jj+1)*(2*Jk+1) / (4*M_PI) );
	return A * wigner3j( Ji, Jj, Jk, 0, 0, 0 ) * wigner3j( Ji, Jj, Jk, Mi, Mj, Mk );
}

double realGaunt( int Ji, int Mi, int Jj, int Mj, int Jk, int Mk )
{
	// \todo Considering the change on equation 26, rethink how would be the special case equations and avoid the brute force approach below.
	double sum = 0;
	for ( int M1 = -Ji; M1 <= Ji; M1++ ) 
	{
		for ( int M2 = -Jj; M2 <= Jj; M2++ )
		{
			for ( int M3 = -Jk; M3 <= Jk; M3++ )
			{
				// \todo Reduce the number of iterations by considering the behavior of U.
				double tmp = ( U(Ji,M1,Mi)*U(Jj,M2,Mj)*U(Jk,M3,Mk) ).real();
				if ( tmp != 0 )
					sum += tmp * gaunt( Ji, M1, Jj, M2, Jk, M3 );
			}
		}
	}
	return sum;
}


float tripleint(int l1, int m1, int l2, int m2, int l3, int m3)
{
	return realGaunt(l1,m1,l2,m2,l3,m3);
	// return clebschGordan(l1,l2,l3,m1,m2,m3);
	// return std::sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*PI))
	// 	* wigner3j(l1,l2,l3,m1,m2,m3) * wigner3j(l1,l2,l3,0,0,0);
}

int SHIndex(int l, int m) {
	return l*l+l+m;
}

int main(int argc, char* argv[])
{
	if (argc<=1) {
		std::cerr << "Usage: " << argv[0] << " <n>\n";
		return 1;
	}
	const int n = atoi(argv[1]);
	std::cout.precision(10);
	for (int l1 = 0; l1 < n; ++l1)
	for (int m1 = -l1; m1 <= l1; ++m1)
	for (int l2 = 0; l2 < n; ++l2)
	for (int m2 = -l2; m2 <= l2; ++m2)
	for (int l3 = 0; l3 < n; ++l3)
	for (int m3 = -l3; m3 <= l3; ++m3)
	{
		int i = SHIndex(l1,m1);
		int j = SHIndex(l2,m2);
		int k = SHIndex(l3,m3);
		auto val = tripleint(l1,m1,l2,m2,l3,m3);
		if (val!=0)
			std::cout << "{" << i << "," << j << "," << k << "," << val << "},\n";
	}
}
