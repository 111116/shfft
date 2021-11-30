#include <iostream>
#include "gen_common.hpp"

// defines conversion from fourier series to SH basis

FS fs_column(const FS& f, int m)
{
	FS s;
	for (int b=-maxdeg+1; b<maxdeg; ++b)
		s.at(b,0) += f.at(b,m);
	return s;
}

complex int0pi(const FS& f)
{
	complex s(0);
	for (int k=-maxdeg+1; k<maxdeg; ++k)
		if ((k%2+2)%2==1)
			s += complex(0,2.0/k) * f.at(k,0);
	s += complex(PI,0) * f.at(0,0);
	return s;
}

complex fs2sh(const FS& fs, int l, int m)
{
	bool neg = false;
	if (m<0) {
		neg = true;
		m = -m;
	}
	static const FS fs_sin_theta = FS(1,0,complex(0,-0.5)) - FS(-1,0,complex(0,-0.5));
	if (m==0) return complex(2*PI*K(l,m))*int0pi(fs_column(fs,m)*fs_P(l,m)*fs_sin_theta);
	if (!neg) return complex(std::sqrt(2)*PI*K(l,m),0)*int0pi((fs_column(fs,m) + fs_column(fs,-m))*fs_P(l,m)*fs_sin_theta);
			  return complex(0,std::sqrt(2)*PI*K(l,m))*int0pi((fs_column(fs,m) - fs_column(fs,-m))*fs_P(l,m)*fs_sin_theta);
}


int main()
{
	fs_P_init();
	int n_mul = 0;
	using std::cout;
	cout.precision(10);
	cout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
	cout << "// Fourier series (up to order " << 2*n-1 << ") to SH (order " << n << ") conversion\n";
	for (int l=0; l<n; ++l)
		for (int m=-l; m<=l; ++m)
		{
			printf("SHreg[%d] = ", SHIndex(l,m));
			int printcnt = 0;
			// symmetry optimization
			for (int a=-(2*n-2); a<=0; ++a)
				// for (int b=-(2*n-2); b<=0; ++b)
				{
					int b = -std::abs(m); // observation: |m| = |b|
					FS fs;
					fs.at(a,b) = 1;
					complex t = fs2sh(fs,l,m);
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
					// thresholding
					if (std::abs(t.real()) < 1e-8)
						t = complex(0, t.imag());
					if (std::abs(t.imag()) < 1e-8)
						t = complex(t.real(), 0);
					// print
					if (t.real() != complex(0)) {
						// if (std::abs(m) != std::abs(b)) std::cerr << "|m| != |b|\n";
						if (printcnt > 0)
							cout << " + ";
						cout << t.real() << " * ";
						printf("FSreg[%d].x", FSIndex(a,b,2*n-1));
						printcnt++;
						n_mul++;
					}
					if (t.imag() != complex(0)) {
						// if (std::abs(m) != std::abs(b)) std::cerr << "|m| != |b|\n";
						// std::cerr << l << " " << m << " " << a << " " << b << "\n";
						cout << " - ";
						cout << t.imag() << " * ";
						printf("FSreg[%d].y", FSIndex(a,b,2*n-1));
						printcnt++;
						n_mul++;
					}
				}
			if (printcnt == 0)
				cout << "0";
			cout << ";\n";
		}
	std::cerr << "fs2sh n_mul = " << n_mul << std::endl;
}
