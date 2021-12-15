// This program is a proof-of-concept demonstration of computing SH products
// with arbitary k value, using the divide-and-conquer strategy. To obtain
// better performance, order of computation may be arranged with care,
// especially for small fixed k value.

#include <iostream>
#include <cmath>
#include "fourierseries.hpp"
#include "fastmul.hpp"
#include "sh.hpp"
#include "shproduct.hpp"
#include "consolelog.hpp"
#include "shorder.hpp"

#include "precompute.hpp"


FourierSeries<n> sh2fs(SH<n> sh)
{
    FourierSeries<n> fs;
    #include "sh2fs.cpp"
    return fs;
}


int main()
{
    // n: order of SH (i.e. n^2 coefficients each SH)
    // Result is (k-1) SH functions multiplied, reprojected onto n-th SH basis.
    int n=4, k=7;
    // Note: n and k can be assigned arbitarily at runtime.
    //       Here we demonstrate runtime precomputation.
    //       Performance of this demo may be suboptimal.
    console.log("n=", n);
    console.log("k=", k);
    precompute_sh2fs(n);
    // Note: if k is too large (i.e. product of many terms is required),
    //       it's usually reasonable to truncate intermediate results
    //       to avoid precomputing a very large conversion for the final result.
    precompute_fs2sh(n,k);

    // random init
    SH<n> input[k-1];
    for (int i=0; i<k-1; ++i)
        for (int l=0; l<n; ++l)
            for (int m=-l; m<=l; m++) {
                input[i].at(l,m) = (float)rand()/RAND_MAX-0.5;
            }
    
    // compute using traditional method

    // Note: it's also viable to apply the divide-and-conquer strategy to traditional method,
    //       but we don't bother it here since we don't compare the performance in this demo.
    //       In other demos where the traditional method is benchmarked against our method,
    //       we used same strategy between two methods, and optimized the traditional method
    //       by removing as much redundant calculation as possible, in favor of the traditional method.

    // Note: When calculating the product, using up to order [n*ceil((k-1)/2)] is sufficient
    //       if the result is projected onto n-th order SH.
    


    // compute using our method (with divide-and-conquer strategy)
    FourierSeries<n> fs[k-1];
    for (int i=0; i<k-1; ++i)
        fs[i] = sh2fs(input[i]);
    auto prod_fs = prod(k-1, fs);


    // validate
    
    console.log("Er:",(sh3-sh4).magnitude() / sh3.magnitude());
}
