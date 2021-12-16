# SHFFT

This repository is the source code of

>  Hanggao Xin, Zhiqian Zhou, Di An, Ling-Qi Yan, Kun Xu, Shi-Min Hu, Shing-Tung Yau. **Fast and Accurate Spherical Harmonics Products.** ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2021)

We provide a demo program in `cpu_general`, as well as both CPU and CUDA version of test programs we used for benchmarking, comparing our method to the traditional method. In these programs, the 𝒌-multiple product operator (⊗𝑘) of 𝒏-th ordered SH is implemented. Refer to the [paper](https://sites.cs.ucsb.edu/~lingqi/publications/paper_shmultprod.pdf) for detailed definition.

## Prerequisites

CPU version: install **Intel MKL** and make sure `MKLROOT` in Makefile points to the installation folder.

GPU version: CUDA is required.

## Build & Usage

```bash
git clone https://github.com/111116/shfft
cd shfft
cd cpu_general # can be cpu_general, cpu_k*, gpu_k*
make
./test
```

## Directory Structure

- `cpu_general`: Our program demonstrating the proposed algorithm. 𝒏 and 𝒌 are determined at runtime.
- `cpu_k*`: Our CPU-version benchmarking program. 𝒏 is determined at compile time; 𝒌 is fixed.
- `gpu_k*`: Our GPU-version benchmarking program. 𝒏 is determined at compile time; 𝒌 is fixed.
- `gamma_bin`: precomputed SH tripling tensor used by the traditional method.

## Algorithm Overview

To recap the paper, our algorithm consists of three steps:

1. Conversion from SH space to Fourier space
2. Convolution in Fourier space 
3. Conversion back to SH space

The second step, 2D convolution, can be accelarated using FFT. We here use the Intel MKL library (FFTW interface) or the cuFFT library. In case there are many inputs, a divide-and-conquer strategy can be used.

The first and third step are essentially sparse linear transforms, of which coefficients can be precomputed. In `cpu_general`, precomputation of our algorithm is done at runtime. In `cpu_k*`, `gpu_k*`, precomputation of our algorithm is done when building the program. 

## Cite Paper

```plain
@article{Xin2021FastAA,
  title={Fast and accurate spherical harmonics products},
  author={Hanggao Xin and Zhiqian Zhou and Di An and Ling-Qi Yan and Kun Xu and Shi-Min Hu and Shing-Tung Yau},
  journal={ACM Transactions on Graphics (TOG)},
  year={2021},
  volume={40},
  pages={1 - 14}
}
```

