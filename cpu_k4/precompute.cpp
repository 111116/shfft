#include <iostream>
#include <fstream>
#include <cmath>
#include "fourierseries.hpp"
#include "legendre.hpp"
#include "shorder.hpp"

const float PI = acos(-1);

float K(int l, int m)
{
	return std::sqrt(float(2*l+1)*fac(l-m)/(4*PI*fac(l+m)));
}

// defines conversion from SH basis to fourier series

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

FS fs_y(int l, int m) {
	if (m == 0) return K(l,m) * fs_P(l,m);
	if (m > 0) return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0.5,0))+FS(0,-m,complex(0.5,0)))*fs_P(l,m);
	m = -m;    return std::sqrt(2.f)*K(l,m)*(FS(0,m,complex(0,-0.5))-FS(0,-m,complex(0,-0.5)))*fs_P(l,m);
}

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


static int SHIndex(int l, int m) {
	return l*l+l+m;
}

int main()
{
	fs_P_init();
	// validation of conversion
	{
		for (int a=0; a<n; ++a)
			for (int b=-a; b<=a; ++b) {
				for (int l=0; l<n; ++l)
					for (int m=-l; m<=l; ++m)
					{
						complex t = fs2sh(fs_y(a,b),l,m);
						if (abs(t)<1e-6) std::cout << "- ";
						else std::cout << t << " ";
					}
				std::cout << "\n";
			}
	}
	// output precomputed conversion matrix
	// SH to FS
	{
		int n_mul = 0;
		std::ofstream fout("sh2fs.cpp");
		fout.precision(10);
		fout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
		fout << "// SH (order " << n << ") to Fourier series conversion\n";
		FS fs[n*n];
		for (int l=0; l<n; ++l)
			for (int m=-l; m<=l; ++m)
				fs[SHIndex(l,m)] = fs_y(l,m);
		for (int a=-n+1; a<=n-1; ++a)
			for (int b=-n+1; b<=n-1; ++b)
			{
				fout << "fs.at(" << a << "," << b << ") = complex( ";
				#define PRINTCOEFF(func)\
				{\
					int printcnt = 0;\
					for (int l=0; l<n; ++l)\
						for (int m=-l; m<=l; ++m)\
							if (func(fs[SHIndex(l,m)].at(a,b)) != complex(0,0)) \
							{\
								if (printcnt > 0)\
									fout << " + ";\
								fout << func(fs[SHIndex(l,m)].at(a,b)) << " * sh.at(" << l << "," << m << ")";\
								printcnt++;\
								n_mul++;\
							}\
					if (printcnt == 0)\
						fout << "0";\
				}
				PRINTCOEFF(std::real);
				fout << ", ";
				PRINTCOEFF(std::imag);
				fout << " );\n";
			}
		std::cerr << "sh2fs n_mul = " << n_mul << std::endl;
	}
	// FS to SH
	{
		int n_mul = 0;
		std::ofstream fout("fs2sh.cpp");
		fout.precision(10);
		fout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
		fout << "// Fourier series (up to order " << 2*n-1 << ") to SH (order " << n << ") conversion\n";
		for (int l=0; l<n; ++l)
			for (int m=-l; m<=l; ++m)
			{
				fout << "sh.at(" << l << "," << m << ") = ";
				int printcnt = 0;
				// symmetry optimization
				for (int a=-(2*n-2); a<=0; ++a)
					for (int b=-(2*n-2); b<=0; ++b)
					{
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
						if (std::abs(t.real()) < 1e-7)
							t = complex(0, t.imag());
						if (std::abs(t.imag()) < 1e-7)
							t = complex(t.real(), 0);
						// print
						if (t.real() != complex(0)) {
							if (printcnt > 0)
								fout << " + ";
							fout << t.real() << " * ";
							fout << "fs.at(" << a << "," << b << ").real()";
							printcnt++;
							n_mul++;
						}
						if (t.imag() != complex(0)) {
							fout << " - ";
							fout << t.imag() << " * ";
							fout << "fs.at(" << a << "," << b << ").imag()";
							printcnt++;
							n_mul++;
						}
					}
				if (printcnt == 0)
					fout << "0";
				fout << ";\n";
			}
		std::cerr << "fs2sh n_mul = " << n_mul << std::endl;
	}


	// FS to SH Mul3
        {
                int n_mul = 0;
                std::ofstream fout("Mul3fs2sh.cpp");
                fout.precision(10);
                fout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
                fout << "// Fourier series (up to order " << 3*n-2 << ") to SH (order " << n << ") conversion\n";
                for (int l=0; l<n; ++l)
                        for (int m=-l; m<=l; ++m)
                        {
                                fout << "sh.at(" << l << "," << m << ") = ";
                                int printcnt = 0;
                                // symmetry optimization
                                for (int a=-(3*n-3); a<=0; ++a)
                                        for (int b=-(3*n-3); b<=0; ++b)
                                        {
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
                                                if (std::abs(t.real()) < 1e-7)
                                                        t = complex(0, t.imag());
                                                if (std::abs(t.imag()) < 1e-7)
                                                        t = complex(t.real(), 0);

                                                // print
                                                if (t.real() != complex(0)) {
                                                        if (printcnt > 0)
                                                                fout << " + ";
                                                        fout << t.real() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").real()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                                if (t.imag() != complex(0)) {
                                                        fout << " - ";
                                                        fout << t.imag() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").imag()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                        }
                                if (printcnt == 0)
                                        fout << "0";
                                fout << ";\n";
                        }
                std::cerr << "3fs2sh n_mul = " << n_mul << std::endl;
        }

	// FS to SH Mul4
	/*
        {
                int n_mul = 0;
                std::ofstream fout("Mul4fs2sh.cpp");
                fout.precision(10);
                fout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
                fout << "// Fourier series (up to order " << 4*n-3 << ") to SH (order " << n << ") conversion\n";
                for (int l=0; l<n; ++l)
                        for (int m=-l; m<=l; ++m)
                        {
                                fout << "sh.at(" << l << "," << m << ") = ";
                                int printcnt = 0;
                                // symmetry optimization
                                for (int a=-(4*n-4); a<=0; ++a)
                                        for (int b=-(4*n-4); b<=0; ++b)
                                        {
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
                                                if (std::abs(t.real()) < 1e-7)
                                                        t = complex(0, t.imag());
                                                if (std::abs(t.imag()) < 1e-7)
                                                        t = complex(t.real(), 0);

                                                // print
                                                if (t.real() != complex(0)) {
                                                        if (printcnt > 0)
                                                                fout << " + ";
                                                        fout << t.real() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").real()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                                if (t.imag() != complex(0)) {
                                                        fout << " - ";
                                                        fout << t.imag() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").imag()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                        }
                                if (printcnt == 0)
                                        fout << "0";
                                fout << ";\n";
                        }
                std::cerr << "4fs2sh n_mul = " << n_mul << std::endl;
        }
	*/
	
	// FS to SH Mul5
	/*
        {
                int n_mul = 0;
                std::ofstream fout("Mul5fs2sh.cpp");
                fout.precision(10);
                fout << "// THIS CODE IS MACHINE-GENERATED. DO NOT EDIT!\n";
                fout << "// Fourier series (up to order " << 5*n-4 << ") to SH (order " << n << ") conversion\n";
                for (int l=0; l<n; ++l)
                        for (int m=-l; m<=l; ++m)
                        {
                                fout << "sh.at(" << l << "," << m << ") = ";
                                int printcnt = 0;
                                // symmetry optimization
                                for (int a=-(5*n-5); a<=0; ++a)
                                        for (int b=-(5*n-5); b<=0; ++b)
                                        {
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
                                                if (std::abs(t.real()) < 1e-7)
                                                        t = complex(0, t.imag());
                                                if (std::abs(t.imag()) < 1e-7)
                                                        t = complex(t.real(), 0);

                                                // print
                                                if (t.real() != complex(0)) {
                                                        if (printcnt > 0)
                                                                fout << " + ";
                                                        fout << t.real() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").real()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                                if (t.imag() != complex(0)) {
                                                        fout << " - ";
                                                        fout << t.imag() << " * ";
                                                        fout << "fs.at(" << a << "," << b << ").imag()";
                                                        printcnt++;
                                                        n_mul++;
                                                }
                                        }
                                if (printcnt == 0)
                                        fout << "0";
                                fout << ";\n";
                        }
                std::cerr << "5fs2sh n_mul = " << n_mul << std::endl;
        }
	*/
}
