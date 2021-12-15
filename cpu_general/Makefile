CXXFLAGS = -std=c++11 -O3 -ffast-math -I..
MKLROOT = /opt/intel/oneapi/mkl/latest
MKLFLAGS = -m64 -I${MKLROOT}/include
MKLLINKS = ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl

test: test.cpp  fourierseries.hpp fastmul.hpp precompute_sh2fs.hpp precompute_fs2sh.hpp fourierseries.hpp legendre.hpp
	$(CXX) $(CXXFLAGS) $< $(MKLFLAGS) $(MKLLINKS) -o $@

