#include <iostream>
#include <cmath>
#include "fourierseries.hpp"
#include "fastmul.hpp"
#include "sh.hpp"
#include "shproduct.hpp"
#include "consolelog.hpp"
#include "shorder.hpp"
#include <ctime>
#include <vector>

FourierSeries<n> sh2fs(SH<n> sh)
{
	FourierSeries<n> fs;
	#include "sh2fs.cpp"
	return fs;
}

SH<n> fs2sh(FourierSeries<2*n-1> fs)
{
	SH<n> sh;
	#include "fs2sh.cpp"
	return sh;
}

SH<n> fs2sh(FourierSeries<3*n-2> fs)
{
	SH<n> sh;
	#include "Mul3fs2sh.cpp"
	return sh;
}


constexpr int M = 2 * n - 1;

SH<n> trunc(SH<M> a)
{
 	SH<n> b;
 	// truncate
 	for (int l=0; l<n; ++l)
  	for (int m=-l; m<=l; m++)
   		b.at(l,m) = a.at(l,m);
	return b;
}

int main()
{
	console.log("n=", n);
	// random init
	SH<M> sh1, sh2, sh3;
 	for (int l=0; l<n; ++l)
	  	for (int m=-l; m<=l; m++) {
	   		sh1.at(l,m) = (float)rand()/RAND_MAX-0.5;
	   		sh2.at(l,m) = (float)rand()/RAND_MAX-0.5;
	   		sh3.at(l,m) = (float)rand()/RAND_MAX-0.5;
	  	}
 	SH<M> ref;
	ref = sh1*sh2*sh3;
	
	SH<n> sh5trunc;
	console.time("trad_trunc");
	for (int i=0; i<10000; ++i)
		sh5trunc = trunc(sh1) * trunc(sh2) * trunc(sh3);
	console.timeEnd("trad_trunc");

	SH<n> sh6;
	console.time("ours");
        for (int i=0; i<10000; ++i)
		sh6 = fs2sh(fastmul(sh2fs(trunc(sh1)),sh2fs(trunc(sh2)),sh2fs(trunc(sh3))));
	console.timeEnd("ours");

	console.log("ours err:",(sh6-trunc(ref)).magnitude() / ref.magnitude());
	console.log("trad err:",(sh6-sh5trunc).magnitude() / sh6.magnitude());
}
