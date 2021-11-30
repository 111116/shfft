CXXFLAGS = -std=c++14 -O3

test : test.cu generated/sh2fs.cu generated/fs2sh.cu
	nvcc $(CXXFLAGS) -I.. $< -lcurand -lcufft -o $@

generated/sh2fs.cu : generated/sh2fs_gen
	./$< > $@

generated/fs2sh.cu : generated/fs2sh_gen
	./$< > $@

generated/sh2fs_gen : generator/sh2fs_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@

generated/fs2sh_gen : generator/fs2sh_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@
