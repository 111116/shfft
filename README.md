# SHFFT

This repository is the source code of

>  Hanggao Xin, Zhiqian Zhou, Di An, Ling-Qi Yan, Kun Xu, Shi-Min Hu, Shing-Tung Yau. **Fast and Accurate Spherical Harmonics Products.** ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2021)

We provide a demo program in `cpu_general`, as well as both CPU and CUDA version of test programs we used for benchmarking, comparing our method to the traditional method. In these programs, the 𝒌-multiple product operator (⊗𝑘) of 𝒏-th ordered SH is implemented. Refer to the paper for detailed definition.

## Prerequisites

CPU version: install **Intel MKL** and make sure `MKLROOT` in Makefile points to the installation folder.

GPU version: CUDA is required

## Build & Usage

```bash
git clone https://github.com/111116/shfft
cd shfft
# cpu_general, cpu_k*, gpu_k*
cd cpu_general
make
./test
```

