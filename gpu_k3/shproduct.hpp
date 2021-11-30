#pragma once

#include <fstream>
#include <iostream>
#include <cstring>
#include "sh.hpp"

template <int n>
matrix<n> operator* (const matrix<n>& a, const matrix<n>& b)
{
    matrix<n> c;
    memset(c.a, 0, sizeof c.a);
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j)
            for (int k=0; k<n; ++k)
                c.a[i][j] += a.a[i][k] * b.a[k][j];
    return c;
}

// projection of product of SH projected functions
template <int n>
SH<n> operator*(const SH<n>& a, const SH<n>& b)
{
    SH<n> c;
    for (auto e: SH<n>::SparseGamma)
        c.a[e.c] += e.val * a.a[e.a] * b.a[e.b];
    return c;
}

// template <int n>
// SH<n> SH<n>::squared()
// {
//     SH<n> c;
//     for (auto e: SH<n>::SquareSparseGamma)
//         c.a[e.c] += e.val * a[e.a] * a[e.b];
//     return c;
// }


template <int n>
float dot(const SH<n>& a, const SH<n>& b)
{
    float t = 0;
    for (int i=0; i<n*n; ++i)
        t += a.a[i] * b.a[i];
    return t;
}


// projection of product as transformation lambda B: A*B
template <int n>
matrix<n*n> SH<n>::prodMatrix() const
{
    matrix<n*n> m;
    for (int i=0; i<n*n; ++i)
    for (int j=0; j<n*n; ++j)
    for (int k=0; k<n*n; ++k)
        m.a[i][j] += a[k] * SH<n>::Gamma[i][j][k];
    return m;
}


template <int n>
SH<n> operator*(const matrix<n*n>& a, const SH<n>& b)
{
    SH<n> c;
    for (int i=0; i<n*n; ++i)
        for (int j=0; j<n*n; ++j)
            c.a[i] += a.a[i][j] * b.a[j];
    return c;
}
