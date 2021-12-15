#include <iostream>
#include "legendre.hpp"

// defines conversion from SH basis to fourier series

namespace precompute {

typedef std::complex<float> res_type;

FS fs_y(int l, int m) {
	if (m == 0) return K(l,m) * P_cosx(l,m);
	if (m > 0) return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0.5,0))+FS(0,-m,complex(0.5,0)))*P_cosx(l,m);
	m = -m;    return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0,-0.5))-FS(0,-m,complex(0,-0.5)))*P_cosx(l,m);
}

std::vector<res_type> coeff_sh2fs;

void precompute_sh2fs(int n)
{
	coeff_sh2fs = std::vector<res_type>(n*n*4*n, 0);
	for (int l=0; l<n; ++l)
		for (int m=-l; m<=l; ++m) {
			const FS fs = fs_y(l,m);
			for (int a=-n+1; a<n; ++a) {
				coeff_sh2fs[SHIndex(l,m)*4*n + a+n-1] = res_type(fs.at(a,-m));
				if (m!=0)
				coeff_sh2fs[SHIndex(l,m)*4*n + a+n-1 + 2*n] = res_type(fs.at(a,m));
			}
		}
}

} // end namespace precompute

using precompute::coeff_sh2fs;
using precompute::precompute_sh2fs;
