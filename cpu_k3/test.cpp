#include <iostream>
#include <cmath>
#include "fourierseries.hpp"
#include "fastmul.hpp"
#include "sh.hpp"
#include "shproduct.hpp"
#include "lib/consolelog.hpp"
#include "shorder.hpp"

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

int main()
{
	console.log("n=", n);
	// random init
	SH<n> sh1, sh2;
	for (int l=0; l<n; ++l)
		for (int m=-l; m<=l; m++) {
			sh1.at(l,m) = (float)rand()/RAND_MAX-0.5;
			sh2.at(l,m) = (float)rand()/RAND_MAX-0.5;
		}
	SH<n> sh3 = sh1*sh2;

	console.time("trad");
	for (int i=0; i<100000; ++i)
		sh3 = sh1*sh2;
	console.timeEnd("trad");

	SH<n> sh4;
	console.time("ours");
	for (int i=0; i<100000; ++i)
		sh4 = fs2sh(fastmul(sh2fs(sh1),sh2fs(sh2)));
	console.timeEnd("ours");
	
	console.log("Er:",(sh3-sh4).magnitude() / sh3.magnitude());
	// {
	// 	FourierSeries<n> a,b;
	// 	a = sh2fs(sh1);
	// 	b = sh2fs(sh2);
	// 	FourierSeries<2*n-1> c;
	// 	console.time("fsmul");
	// 	for (int i=0; i<100000; ++i)
	// 		c = fastmul(a,b);
	// 	console.timeEnd("fsmul");
	// }
}
