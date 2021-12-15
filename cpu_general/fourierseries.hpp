// complex Fourier series: sum_{a,b} f(a,b)e^i(ax+by)
#pragma once

#include <complex>
#include <vector>

template <typename T>
struct FourierSeries
{
	typedef std::complex<T> complex;
	std::vector<complex> a; // coefficient; [-n+1..n-1]x[-n+1..n-1]
	int n;
	FourierSeries(): n(1), a(1,0) {}
	FourierSeries(complex c): n(1), a(1,c) {}
	FourierSeries(int i, int j, complex c)
	{
		n = std::max(abs(i)+1, abs(j)+1);
		a = std::vector<complex>((2*n-1)*(2*n-1), 0);
		at(i,j) = c;
	}
	complex& at(int i, int j)
	{
		if (abs(i)>=n || abs(j)>=n) throw "out of range";
		return a[(i+n-1)*(2*n-1)+j+n-1];
	}
	// WARNING: this overload won't be matched if not const
	complex at(int i, int j) const
	{
		if (abs(i)>=n || abs(j)>=n) return 0;
		return a[(i+n-1)*(2*n-1)+j+n-1];
	}
	complex eval(T x, T y) const
	{
		complex sum(0);
		for (int i=-n+1; i<n; ++i)
		for (int j=-n+1; j<n; ++j)
			sum += at(i,j) * std::exp(complex(0,i*x+j*y));
		return sum;
	}
	static FourierSeries zeros(int n) {
		FourierSeries a;
		a.n = n;
		a.a = std::vector<complex>((2*n-1)*(2*n-1), 0);
		return a;
	}
};

template <typename T>
FourierSeries<T> operator+ (const FourierSeries<T>& a, const FourierSeries<T>& b)
{
	int n = std::max(a.n, b.n);
	FourierSeries<T> c = FourierSeries<T>::zeros(n);
	for (int i=-n+1; i<=n-1; ++i)
	for (int j=-n+1; j<=n-1; ++j)
	{
		c.at(i,j) = a.at(i,j) + b.at(i,j);
	}
	return c;
}

template <typename T>
FourierSeries<T>& operator+= (FourierSeries<T>& a, FourierSeries<T> b)
{
	a = a + b;
	return a;
}


template <typename T>
FourierSeries<T> operator- (const FourierSeries<T>& a, const FourierSeries<T>& b)
{
	int n = std::max(a.n, b.n);
	FourierSeries<T> c = FourierSeries<T>::zeros(n);
	for (int i=-n+1; i<=n-1; ++i)
	for (int j=-n+1; j<=n-1; ++j)
	{
		c.at(i,j) = a.at(i,j) - b.at(i,j);
	}
	return c;
}

template <typename T>
FourierSeries<T>& operator-= (FourierSeries<T>& a, const FourierSeries<T>& b)
{
	a = a - b;
	return a;
}

template <typename T>
FourierSeries<T> operator* (std::complex<T> k, const FourierSeries<T>& b)
{
	FourierSeries<T> c=b;
	for (auto& val: c.a)
		val *= k;
	return c;
}

template <typename T>
FourierSeries<T> operator* (T k, const FourierSeries<T>& b)
{
	FourierSeries<T> c=b;
	for (auto& val: c.a)
		val *= k;
	return c;
}

template <typename T>
FourierSeries<T> operator- (const FourierSeries<T>& a)
{
	return std::complex<T>(-1,0) * a;
}

template <typename T>
FourierSeries<T> operator* (const FourierSeries<T>& a, const FourierSeries<T>& b)
{
	FourierSeries<T> c = FourierSeries<T>::zeros(a.n + b.n - 1);
	for (int i1=-a.n+1; i1<a.n; ++i1)
	for (int j1=-a.n+1; j1<a.n; ++j1)
		if (a.at(i1,j1)!=std::complex<T>(0,0))
			for (int i2=-b.n+1; i2<b.n; ++i2)
			for (int j2=-b.n+1; j2<b.n; ++j2)
				c.at(i1+i2, j1+j2) += a.at(i1,j1) * b.at(i2,j2);
	return c;
}

template <typename T>
FourierSeries<T> pow (const FourierSeries<T>& a, int k)
{
	if (k<0) throw "unimplemented";
	FourierSeries<T> prod(0,0,std::complex<T>(1,0));
	for (int i=0; i<k; ++i)
		prod = prod * a;
	return prod;
}


