#include <iostream>
#include "legendre.hpp"

// defines conversion from fourier series to SH basis

namespace precompute {

typedef std::complex<float> res_type;

FS fs_column(const FS& f, int m)
{
	FS s = FS::zeros(f.n);
	for (int b=-f.n+1; b<f.n; ++b)
		s.at(b,0) += f.at(b,m);
	return s;
}

// compute ∫_0^π (sum_a f(a,0) e^iaθ) dθ
complex int0pi(const FS& f)
{
	complex s(0);
	for (int k=-f.n+1; k<f.n; ++k)
		if ((k%2+2)%2==1)
			s += complex(0,2.0/k) * f.at(k,0);
	s += complex(PI,0) * f.at(0,0);
	return s;
}

// compute ∫∫ (sum_{a,b} f(a,b) e^i(aθ+bφ)) y_l^m(θ,φ) dφ sinθ dθ
complex compute_fs2sh(const FS& fs, int l, int m)
{
	bool neg = false;
	if (m<0) {
		neg = true;
		m = -m;
	}
	static const FS fs_sin_theta = FS(1,0,complex(0,-0.5)) - FS(-1,0,complex(0,-0.5));
	if (m==0) return complex(2*PI*K(l,m))*int0pi(fs_column(fs,m)*P_cosx(l,m)*fs_sin_theta);
	if (!neg) return complex(std::sqrt(2)*PI*K(l,m),0)*int0pi((fs_column(fs,m) + fs_column(fs,-m))*P_cosx(l,m)*fs_sin_theta);
			  return complex(0,std::sqrt(2)*PI*K(l,m))*int0pi((fs_column(fs,m) - fs_column(fs,-m))*P_cosx(l,m)*fs_sin_theta);
}

std::vector<res_type> coeff_fs2sh;

// assuming input FS can be represented by SH, thus symmetric
void precompute_fs2sh(int n, int k)
{
	coeff_fs2sh = std::vector<res_type>(n*n*k*n);
	for (int l=0; l<n; ++l)
		for (int m=-l; m<=l; ++m)
		{
			// symmetry optimization
			for (int a=-(k*n-k); a<=0; ++a)
			{
				int b = -std::abs(m);
				FS fs(a,b,1);
				complex t = compute_fs2sh(fs,l,m);
				// symmetry optimization
				if (a<0) t *= 2;
				if (b<0) {
					if ((m < 0) ^ (b%2==0)) {
						t = complex(2*t.real(), 0);
					}
					else {
						t = complex(0, 2*t.imag());
					}
				}
				coeff_fs2sh[SHIndex(l,m) * k*n -a] = res_type(t);
			}
		}
}

} // end namespace precompute

using precompute::coeff_fs2sh;
using precompute::precompute_fs2sh;

