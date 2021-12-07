#pragma once

#include <cstring>
#include <complex>
#include <iosfwd>

typedef std::complex<float> complex;

template <int n>
struct LargeFourierSeries
{
	static const unsigned size = (2*n-1)*(2*n-1);
	complex* a = NULL;
	LargeFourierSeries()
	{
		a = new complex[size];
		memset(a, 0, size*sizeof(complex));
	}
	~LargeFourierSeries()
	{
		delete[] a;
	}
	LargeFourierSeries(int i, int j, complex c): LargeFourierSeries()
	{
		if (abs(i)>=n || abs(j)>=n) throw "out of range";
		at(i,j) = c;
	}
	LargeFourierSeries(const LargeFourierSeries& b): LargeFourierSeries()
	{
		memcpy(a, b.a, size*sizeof(complex));
	}
	LargeFourierSeries(LargeFourierSeries&& b)
	{
		a = b.a;
		b.a = NULL;
	}
	LargeFourierSeries& operator=(LargeFourierSeries&& b)
	{
		delete[] a;
		a = b.a;
		b.a = NULL;
		return *this;
	}
	LargeFourierSeries& operator=(const LargeFourierSeries& b)
	{
		memcpy(a, b.a, size*sizeof(complex));
		return *this;
	}
	complex& at(int i, int j)
	{
		return a[(i+n-1)*(2*n-1)+(j+n-1)];
	}
	complex const& at(int i, int j) const
	{
		return a[(i+n-1)*(2*n-1)+(j+n-1)];
	}
	complex eval(float x, float y) const
	{
		complex sum(0);
		for (int i=-n+1; i<n; ++i)
		for (int j=-n+1; j<n; ++j)
			sum += at(i,j) * std::exp(complex(0,i*x+j*y));
		return sum;
	}
};

template <int n>
LargeFourierSeries<n> operator+ (LargeFourierSeries<n> a, LargeFourierSeries<n> b)
{
	LargeFourierSeries<n> c;
	for (int i=0; i<LargeFourierSeries<n>::size; ++i)
		c.a[i] = a.a[i] + b.a[i];
	return c;
}

template <int n>
LargeFourierSeries<n>& operator+= (LargeFourierSeries<n>& a, LargeFourierSeries<n> b)
{
	a = a + b;
	return a;
}

template <int n>
LargeFourierSeries<n> operator- (LargeFourierSeries<n> a, LargeFourierSeries<n> b)
{
	LargeFourierSeries<n> c;
	for (int i=0; i<LargeFourierSeries<n>::size; ++i)
		c.a[i] = a.a[i] - b.a[i];
	return c;
}

template <int n>
LargeFourierSeries<n> operator* (complex k, LargeFourierSeries<n> b)
{
	LargeFourierSeries<n> c=b;
	for (int i=0; i<LargeFourierSeries<n>::size; ++i)
		c.a[i] *= k;
	return c;
}

template <int n>
LargeFourierSeries<n> operator- (LargeFourierSeries<n> a)
{
	return complex(-1,0) * a;
}

template <int n>
LargeFourierSeries<n> operator* (LargeFourierSeries<n> a, LargeFourierSeries<n> b)
{
	LargeFourierSeries<n> c;
	for (int i1=-n+1; i1<n; ++i1)
	for (int j1=-n+1; j1<n; ++j1)
	if (a.at(i1,j1)!=complex(0,0))
	for (int i2=-n+1; i2<n; ++i2)
	for (int j2=-n+1; j2<n; ++j2)
	{
		complex t = a.at(i1,j1) * b.at(i2,j2);
		int i = i1+i2;
		int j = j1+j2;
		// if (t != complex(0) and (std::abs(i)>=n or std::abs(j)>=n))
		// 	throw "overflowing";
		if (std::abs(i)<n and std::abs(j)<n)
			c.at(i,j) += t;
	}
	return c;
}

template <int n>
LargeFourierSeries<n> pow (LargeFourierSeries<n> a, int k)
{
	if (k<0) throw "unimplemented";
	LargeFourierSeries<n> prod(0,0,complex(1,0));
	for (int i=0; i<k; ++i)
		prod = prod * a;
	return prod;
}

template <int n>
std::ostream& operator<< (std::ostream& out, const LargeFourierSeries<n>& fs)
{
	out << "[FS] ";
	bool comma = false;
	for (int i=-n+1; i<n; ++i)
	for (int j=-n+1; j<n; ++j)
		if (fs.at(i,j) != complex(0)) {
			if (comma) out << ", ";
			out << "(" << i << "," << j << ")->" << fs.at(i,j);
		}
	return out;	
}

