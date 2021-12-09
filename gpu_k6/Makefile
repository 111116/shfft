CXXFLAGS = -std=c++14 -O3

test : test.cu select_size.hpp generated/sh1_fs5.cu generated/fs5_sh1.cu readgamma.hpp
	nvcc $(CXXFLAGS) -I.. $< -lcurand -lcufft -o $@

generated/sh1_fs5.cu : generated/sh1_fs5_gen
	./$< > $@

generated/fs5_sh1.cu : generated/fs5_sh1_gen
	./$< > $@

generated/sh1_fs3_gen : generator/sh1_fs3_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@

generated/fs3_sh1_gen : generator/fs3_sh1_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@

generated/sh1_fs5_gen : generator/sh1_fs5_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@

generated/fs5_sh1_gen : generator/fs5_sh1_gen.cpp shorder.hpp n
	$(CXX) -I.. -I. $(CXXFLAGS) $< -o $@
