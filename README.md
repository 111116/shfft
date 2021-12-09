# SHFFT

This repository is the source code of

>  Hanggao Xin, Zhiqian Zhou, Di An, Ling-Qi Yan, Kun Xu, Shi-Min Hu, Shing-Tung Yau. **Fast and Accurate Spherical Harmonics Products.** ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2021)

We provide both CPU and CUDA version of the test programs, comparing our method to the traditional method. In these programs, the 𝒌-multiple product operator (⊗𝑘) of 𝒏-th ordered SH is implemented. Refer to the paper for detailed definition.

## Prerequisites

CPU version: install **Intel MKL** and make sure `MKLROOT` in Makefile points to the installation folder.

GPU version: CUDA is required

## Build & Usage

