CXXFLAGS = -std=c++11 -O3 -ffast-math
#CXXFLAGS = -std=c++11 -g

test: test.cpp sh2fs.hpp fs2sh.hpp fourierseries.hpp fastmul.hpp shorder.hpp
	$(CXX) $(CXXFLAGS) $< -lfftw3f -lfftw3 -o $@

precompute: precompute.cpp fourierseries.hpp legendre.hpp shorder.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

sh2fs.hpp: precompute
	./precompute

fs2sh.hpp: precompute
#	./precompute
