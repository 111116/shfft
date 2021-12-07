// class SH: table of spherical harmonics coefficients,
// used to approximate a spherical function

#pragma once

#include <functional>
#include <cmath>
// #include "math/vecmath.hpp"
// #include "math/matmath.hpp"
// #include "image.hpp"

// typedef std::function<float(vec3f)> sphfunc;

const float PI = acos(-1);

template <int n>
struct matrix
{
    // row major
    float a[n][n] = {{0}};
};
template <int n>
matrix<n> operator* (const matrix<n>&, const matrix<n>&);



template <int n>
class SH;

template <int n>
class SymmSH
{
public:
    // coefficients f(l,m=0)
    float a[n] = {0};
    SymmSH(){}
    // from spherical function symmetric around z-axis (about theta)
    SymmSH(std::function<float(float)>, int nsample = 100000);
    // rotated to be centered around arbitrary axis z'
    // SH<n> rotated(vec3f z);
};


struct TensorEntry
{
    short a,b,c;
    float val;
};

template <int n>
class SH
{
	static const int lmax;
    // static const float*** load_triple_product_tensor();
	static int SHIndex(int l, int m) {
		return l*l+l+m;
	}
public:
    static float Gamma[n*n][n*n][n*n];
    static TensorEntry SparseGamma[];
    static TensorEntry SquareSparseGamma[];
    static int placeholder;
    float a[n*n] = {0};
    static const SH unit();

	float& at(int l, int m) {
		return a[SHIndex(l,m)];
	}
    float const& at(int l, int m) const {
        return a[SHIndex(l,m)];
    }
	SH(){}
    SH(SymmSH<n>);
	// projection using sampling
	// SH(sphfunc, int nsample = 1000000);
	// reconstruction
	// float eval(vec3f) const;
	// rotation
	// SH rotated(vec3f);
    // suppress ringing
    SH windowed();
    SH squared();
	// phi-theta visualization
	// Image visualized(int yres = 200);
    // projection of production as transformation M
    matrix<n*n> prodMatrix() const;
    float magnitude() const;
};


template <int n>
SymmSH<n> log(const SymmSH<n>& a);
template <int n>
float dot(const SH<n>&, const SH<n>&);
template <int n>
SH<n> log(const SH<n>&);
template <int n>
SH<n> exp(const SH<n>&);
template <int n>
SH<n> exp_OL(const SH<n>&);
template <int n>
SH<n> exp_HYB(SH<n>);
template <int n>
SH<n> exp_PS(const SH<n>&, int maxk=30);
// To use these functions, include shexp.hpp

template <int n>
SymmSH<n> operator+(const SymmSH<n>& a, const SymmSH<n>& b);
template <int n>
SymmSH<n> operator-(const SymmSH<n>& a, const SymmSH<n>& b);
template <int n>
SymmSH<n> operator*(float k, const SymmSH<n>& b);

template <int n>
SH<n> operator+(const SH<n>& a, const SH<n>& b);
template <int n>
SH<n> operator-(const SH<n>& a, const SH<n>& b);
template <int n>
SH<n> operator*(float k, const SH<n>& b);
template <int n>
SH<n> operator*(const SH<n>& a, const SH<n>& b);
template <int n>
SH<n> operator*(const matrix<n*n>& a, const SH<n>& b);


// ============== implementation starts ==============


// #include "mtsampler.hpp"

// void SHEvaluate(const vec3f &w, int lmax, float *out);
// void SHRotate(const Color *c_in, Color *c_out, const mat4f &m, int lmax);

inline int SHIndex(int l, int m) {
    return l*l+l+m;
}

// #include "shproject.hpp"
// #include "shrotate.hpp"
// #include "shproduct.hpp"
// #include "shlog.hpp"


template <int n>
const int SH<n>::lmax = n-1;

template <int n>
const SH<n> SH<n>::unit()
{
    SH<n> a;
    a.a[0] = std::sqrt(4*PI);
    return a;
}

// convert symmetric to general
template <int n>
SH<n>::SH(SymmSH<n> b)
{
    for (int i=0; i<n; ++i)
        at(i,0) = b.a[i];
}


// windowing
template <int n>
SH<n> SH<n>::windowed()
{
    SH<n> a = *this;
    for (int l=0; l<n; ++l) {
        float alpha = cos(PI/2*(l/(2.0f*n)));
        for (int m = -l; m <= l; ++m)
            a.at(l,m) *= alpha;
    }
    return a;
}


template <int n>
SH<n> operator+(const SH<n>& a, const SH<n>& b)
{
    SH<n> c;
    for (int l=0; l<n; ++l)
        for (int m=-l; m<=l; ++m)
            c.at(l,m) = a.at(l,m) + b.at(l,m);
    return c;
}

template <int n>
SH<n> operator-(const SH<n>& a, const SH<n>& b)
{
    SH<n> c;
    for (int l=0; l<n; ++l)
        for (int m=-l; m<=l; ++m)
            c.at(l,m) = a.at(l,m) - b.at(l,m);
    return c;
}

template <int n>
SH<n> operator*(float k, const SH<n>& b)
{
    SH<n> c = b;
    for (int i=0; i<n*n; ++i)
        c.a[i] *= k;
    return c;
}

template <int n>
float SH<n>::magnitude() const
{
    float t = 0;
    for (int i=0; i<n*n; ++i)
        t += a[i] * a[i];
    return std::sqrt(t);
}


template <int n>
SymmSH<n> operator+(const SymmSH<n>& a, const SymmSH<n>& b)
{
    SymmSH<n> c;
    for (int l=0; l<n; ++l)
        c.a[l] = a.a[l] + b.a[l];
    return c;
}

template <int n>
SymmSH<n> operator-(const SymmSH<n>& a, const SymmSH<n>& b)
{
    SymmSH<n> c;
    for (int l=0; l<n; ++l)
        c.a[l] = a.a[l] - b.a[l];
    return c;
}

template <int n>
SymmSH<n> operator*(float k, const SymmSH<n>& b)
{
    SymmSH<n> c = b;
    for (int i=0; i<n; ++i)
        c.a[i] *= k;
    return c;
}
