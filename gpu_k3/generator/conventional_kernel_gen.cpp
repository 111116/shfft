#include <iostream>
#include <cstdio>
#include <string>
#include "shorder.hpp"
#include "shproduct.hpp"

int main()
{
    for (auto e: SH<n>::SparseGamma)
    {
        // c.a[e.c] += e.val * a.a[e.a] * b.a[e.b];
    	printf("Creg[%d] += %f * Areg[%d] * Breg[%d];\n", e.c, e.val, e.a, e.b);
    }
}
