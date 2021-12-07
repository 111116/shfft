#include <iostream>
#include <cmath>
#include "fourierseries.hpp"
#include "fastmul.hpp"
#include "sh.hpp"
#include "shproduct.hpp"
#include "consolelog.hpp"
#include "shorder.hpp"

FourierSeries<n> sh2fs(SH<n> sh)
{
	FourierSeries<n> fs;
	#include "sh2fs.cpp"
	return fs;
}
/*
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

SH<n> fs2sh(FourierSeries<4*n-3> fs)
{
	SH<n> sh;
	#include "Mul4fs2sh.cpp"
	return sh;
}
*/

SH<n> fs2sh(FourierSeries<5*n-4> fs)
{
        SH<n> sh;
        #include "Mul5fs2sh.cpp"
        return sh;
}

constexpr int M = 4 * n - 3;

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

	// load gamma file
	printf("processing %d\n", n);
	SH<n>::init();
	printf("processing %d\n", M);
	SH<M>::init();

	// random init
	SH<M> sh1, sh2, sh3, sh4, sh5;
 	for (int l=0; l<n; ++l)
	  	for (int m=-l; m<=l; m++) {
	   		sh1.at(l,m) = (float)rand()/RAND_MAX-0.5;
	   		sh2.at(l,m) = (float)rand()/RAND_MAX-0.5;
	   		sh3.at(l,m) = (float)rand()/RAND_MAX-0.5;
			sh4.at(l,m) = (float)rand()/RAND_MAX-0.5;
			sh5.at(l,m) = (float)rand()/RAND_MAX-0.5;
	  	}

  	// reference
 	SH<M> sh6;
	sh6 = sh1*sh2*sh3*sh4*sh5;
	SH<n> ref = trunc(sh6);
	
	// traditional
	SH<n> truncsh6;
	console.time("trad");
	for (int i=0; i<10000; ++i)
		truncsh6 = trunc(sh1) * trunc(sh2) * trunc(sh3) * trunc(sh4) * trunc(sh5);
	console.timeEnd("trad");

	// ours accurate
	SH<n> sh7;
	console.time("ours");
        for (int i=0; i<10000; ++i)
		sh7 = fs2sh(fastmul(sh2fs(trunc(sh1)),sh2fs(trunc(sh2)),sh2fs(trunc(sh3)),sh2fs(trunc(sh4)),sh2fs(trunc(sh5))));
	console.timeEnd("ours");

	console.log("ours err:",(sh7-ref).magnitude() / ref.magnitude());
	console.log("trad err:", (truncsh6-ref).magnitude() / ref.magnitude());
	
}
