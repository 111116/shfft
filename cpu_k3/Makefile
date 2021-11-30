CXXFLAGS = -std=c++11 -O3 -ffast-math -I..
MKLROOT = /opt/intel/oneapi/mkl/latest
MKLFLAGS = -m64 -I${MKLROOT}/include
MKLLINKS = ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl

test: test.cpp sh2fs.cpp fs2sh.cpp fourierseries.hpp fastmul.hpp shorder.hpp
	$(CXX) $(CXXFLAGS) $< $(MKLFLAGS) $(MKLLINKS) -o $@

precompute: precompute.cpp fourierseries.hpp legendre.hpp shorder.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

sh2fs.cpp: precompute
	./precompute

fs2sh.cpp: precompute
	./precompute

