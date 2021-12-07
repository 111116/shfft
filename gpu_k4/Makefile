CXXFLAGS = -std=c++14 -O3

test : test.cu select_size.hpp generated/sh1_fs3.cu generated/fs3_sh1.cu shfft.cu
	nvcc $(CXXFLAGS) -I.. $< -lcurand -lcufft -o $@

generated/sh1_fs3.cu : generated/sh1_fs3_gen
	./$< > $@

generated/fs3_sh1.cu : generated/fs3_sh1_gen
	./$< > $@

generated/sh1_fs3_gen : generator/sh1_fs3_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@

generated/fs3_sh1_gen : generator/fs3_sh1_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@
