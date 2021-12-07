#include <iostream>
#include "gen_common.hpp"

// defines conversion from SH basis to fourier series

FS fs_y(int l, int m) {
	if (m == 0) return K(l,m) * fs_P(l,m);
	if (m > 0) return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0.5,0))+FS(0,-m,complex(0.5,0)))*fs_P(l,m);
	m = -m;    return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0,-0.5))-FS(0,-m,complex(0,-0.5)))*fs_P(l,m);
}

FS fs[n*n];

int main()
{
	fs_P_init();
	int n_mul = 0;
	using std::cout;
	cout.precision(10);
	cout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
	cout << "// SH (order " << n << ") to Fourier series conversion\n";
	for (int l=0; l<n; ++l)
		for (int m=-l; m<=l; ++m)
			fs[SHIndex(l,m)] = fs_y(l,m);
	for (int a=-n+1; a<=n-1; ++a)
		for (int b=-n+1; b<=n-1; ++b)
		{
			#define PRINTCOEFF(func)\
			{\
				int printcnt = 0;\
				for (int l=0; l<n; ++l)\
					for (int m=-l; m<=l; ++m)\
						if (func(fs[SHIndex(l,m)].at(a,b)) != complex(0,0)) \
						{\
							if (printcnt > 0)\
								cout << " + ";\
							cout << func(fs[SHIndex(l,m)].at(a,b)) << " * SHreg[" << SHIndex(l,m) << "]";\
							printcnt++;\
							n_mul++;\
						}\
				if (printcnt == 0)\
					cout << "0";\
			}
			printf("FSreg[%d].x = ", FS3_Index(a,b,n));
			PRINTCOEFF(std::real);
			cout << ";\n";
			printf("FSreg[%d].y = ", FS3_Index(a,b,n));
			PRINTCOEFF(std::imag);
			cout << ";\n";
		}
	std::cerr << "sh2fs n_mul = " << n_mul << std::endl;
}
