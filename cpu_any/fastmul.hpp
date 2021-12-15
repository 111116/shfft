#pragma once

#include <fftw3.h>
#include "fourierseries.hpp"
#include "select_size.hpp"

// EXPLANATION
// multiplication of two 2D Fourier series can be viewed as 2D convolution,
// which can be done with 2D FFT, with a complexity of O(n^2 log n)

// Here we use a naive divide-and-conquer strategy to compute product of multiple inputs.
// The strategy can be significantly optimized by, for example, rearranging multiplications to reduce conversions from one size to another

// template <int n>
// FourierSeries<2*n-1> fastmul (FourierSeries<n> a, FourierSeries<n> b)
// {
//     // startup initialization
//     static const int N = select_size(n);
//     static const int M = 2*n-1;
//     static fftwf_complex* const pool0 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
//     static fftwf_complex* const poolT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
//     static fftwf_complex* const pool1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
//     static fftwf_complex* const pool2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
//     static fftwf_plan p1_1 = fftwf_plan_many_dft(1, &N, M, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
//     static fftwf_plan p1_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool1, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
//     static fftwf_plan p2_1 = fftwf_plan_many_dft(1, &N, M, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
//     static fftwf_plan p2_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool2, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
//     static fftwf_plan p3_1 = fftwf_plan_many_dft(1, &N, N, pool1, NULL, 1, N, pool2, NULL, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
//     // in result we only need column [n-1, 2n-1)
//     static fftwf_plan p3_2 = fftwf_plan_many_dft(1, &N, n, pool2+n-1, NULL, N, 1, pool2+n-1, NULL, N, 1, FFTW_BACKWARD, FFTW_MEASURE);
//     static void* init0once = memset(pool0, 0, N*N*sizeof(fftwf_complex));
//     static void* initTonce = memset(poolT, 0, N*N*sizeof(fftwf_complex));
//     // DFT on coefficients of a
//     for (int i=0; i<2*n-1; ++i)
//         memcpy(pool0+i*N, a.a[i], (2*n-1)*sizeof(fftwf_complex));
//     fftwf_execute(p1_1);
//     fftwf_execute(p1_2);
//     // DFT on coefficients of b, multiplied by coefficient 1/N^2
//     for (int i=0; i<2*n-1; ++i)
//         for (int j=0; j<2*n-1; ++j)
//             *reinterpret_cast<complex*>(pool0+i*N+j) = 1.0f/(N*N) * b.a[i][j];
//     fftwf_execute(p2_1);
//     fftwf_execute(p2_2);
//     // do multiply
//     for (int i=0; i<N*N; ++i)
//         *reinterpret_cast<complex*>(pool1+i) *= *reinterpret_cast<complex*>(pool2+i);
//     // IDFT
//     fftwf_execute(p3_1);
//     fftwf_execute(p3_2);
//     // extract final values
//     FourierSeries<2*n-1> c;
//     for (int i=0; i<4*n-3; ++i)
//         memcpy(c.a[i]+n-1, pool2+i*N+n-1, n*sizeof(fftwf_complex));
//     return c;
// }

// to change float to double, substitute fftwf_ to fftw_
FourierSeries<float> prod_divide_and_conquer(std::vector<FourierSeries<float>> inputs)
{
    typedef std::complex<float> data_t;
    static_assert(sizeof(data_t) == sizeof(FourierSeries<float>::complex), "data type mismatch");
    static_assert(sizeof(data_t) == sizeof(fftwf_complex), "data type mismatch");
    // number of multiplicants
    int n = inputs.size();
    if (n==0) throw "empty input";
    if (n==1) return inputs[0];
    // The divide-and-conquer process used here can be view as follows:
    // Each input corresponds to a leaf node in a complete binary tree.
    // We multiply inputs along the tree from bottom to top.
    int depth = 1; // height of the tree
    for (int t=n-1; t; t>>=1) depth++;
    // dimension of each input element
    for (auto x: inputs)
        if (x.n != inputs[0].n)
            throw "inputs of different size are not handled";
    int datasize[depth];
    datasize[0] = inputs[0].n*2-1;
    for (int i=1; i<depth; ++i)
        datasize[i] = datasize[i-1]*2-1;
    // actual dimension of operated FFT buffer. It's a bit bigger, for performance of FFT
    int fftsize[depth];
    for (int i=0; i<depth; ++i)
        fftsize[i] = bestfftsize(datasize[i]);
    
    // allocate buffers
    // Each buffer, as well as inputs, are square matrices of complex numbers.
    // Since buffers are a bit bigger, data is placed at upper-left corner (lower indices), with the remainder zero-filled.
    // dataflow: pool [copy]-> pool_tmp [DFT]-> pool_tmp_pv [IDFT]-> pool
    fftwf_complex* pool[depth][2];
    fftwf_complex* pool_tmp[depth][2];
    fftwf_complex* pool_tmp_pv[depth][2];
    for (int i=0; i<depth; ++i) {
        int N = fftsize[i];
        for (int j=0; j<=1; ++j) {
            pool       [i][j] = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
            pool_tmp   [i][j] = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
            pool_tmp_pv[i][j] = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        }
    }
    // generate fft plans
    fftwf_plan fft_plan_forward[depth-1][2]; // layer, source
    fftwf_plan fft_plan_inverse[depth-1][2]; // layer, destination
    for (int i=0; i<depth-1; ++i) {
        int N = fftsize[i+1];
        for (int j=0; j<=1; ++j) {
            fft_plan_forward[i][j] = fftwf_plan_dft_2d(N,N, pool_tmp[i+1][j], pool_tmp_pv[i+1][j], FFTW_FORWARD, FFTW_MEASURE);
            fft_plan_inverse[i][j] = fftwf_plan_dft_2d(N,N, pool_tmp_pv[i+1][0], pool[i+1][j], FFTW_BACKWARD, FFTW_MEASURE);
        }
    }

    auto copy_square_data = [](fftwf_complex const* src, int src_size, fftwf_complex* dst, int dst_size)
    {
        int data_size = std::min(src_size, dst_size);
        memset(dst, 0, dst_size * dst_size * sizeof(data_t));
        for (int i=0; i<data_size; ++i)
            memcpy(dst + i * dst_size, src + i * src_size, data_size * sizeof(data_t));
    };
    auto multiply = [&](int layer, int dest) // pool[layer+1] = pool[layer][0] x pool[layer][1]
    {
        // promote the size of input data, then run DFT to get point values
        copy_square_data(pool[layer][0], fftsize[layer], pool_tmp[layer+1][0], fftsize[layer+1]);
        copy_square_data(pool[layer][1], fftsize[layer], pool_tmp[layer+1][1], fftsize[layer+1]);
        fftwf_execute(fft_plan_forward[layer][0]);
        fftwf_execute(fft_plan_forward[layer][1]);
        // multiply point values, divided by size of DFT
        int N = fftsize[layer+1];
        for (int i=0; i<N*N; ++i)
            *reinterpret_cast<data_t*>(pool_tmp_pv[layer+1][0]+i) *= 1.f/(N*N) * *reinterpret_cast<data_t*>(pool_tmp_pv[layer+1][1]+i);
        // IDFT
        fftwf_execute(fft_plan_inverse[layer][dest]);
        // console.log("--", reinterpret_cast<data_t>(*(pool[layer][0])));
        // console.log("--", reinterpret_cast<data_t>(*(pool[layer][1])));
        // console.log("--", reinterpret_cast<data_t>(*(pool[layer+1][0])));
    };
    auto copy_from_input = [copy_square_data](FourierSeries<float> const& input, fftwf_complex* pool, int dst_size)
    {
        copy_square_data(reinterpret_cast<fftwf_complex const*>(input.a.data()), input.n*2-1, pool, dst_size);
    };
    auto copy_to_output = [copy_square_data](FourierSeries<float>& output, fftwf_complex const* pool, int src_size, int src_data_size)
    {
        copy_square_data(pool, src_size, reinterpret_cast<fftwf_complex*>(output.a.data()), output.n*2-1);
    };
    // number of per-layer buffer occupied
    int layer_state[depth];
    for (int i=0; i<depth; ++i)
        layer_state[i] = 0;
    // Actual computation starts here. First we add in all inputs
    for (int id = 0; id < n; ++id) {
        copy_from_input(inputs[id], pool[0][layer_state[0]], fftsize[0]);
        int c = 0; // currently visiting layer
        // if both buffers are occupied now, we need to merge from bottom to top
        while (layer_state[c] == 1) {
            int dest = layer_state[c+1];
            multiply(c, dest);
            layer_state[c] = 0;
            c += 1;
        }
        layer_state[c] = 1;
    }
    // then merge in all remaining nodes
    for (int c = 0; c < depth-1; ++c) {
        if (layer_state[c] == 1) {
            int dest = layer_state[c+1];
            copy_square_data(pool[c][0], fftsize[c], pool[c+1][dest], fftsize[c+1]);
            layer_state[c] = 0;
            layer_state[c+1] += 1;
        }
        if (layer_state[c] == 2) {
            int dest = layer_state[c+1];
            multiply(c, dest);
            layer_state[c] = 0;
            layer_state[c+1] += 1;
        }
    }
    // produce result
    FourierSeries<float> output = FourierSeries<float>::zeros(n * (inputs[0].n - 1) + 1);
    copy_to_output(output, pool[depth-1][0], fftsize[depth-1], datasize[depth-1]);
    // dealloc. These resources shall be reused if this function is to be called multiple times
    for (int i=0; i<depth-1; ++i) {
        for (int j=0; j<=1; ++j) {
            fftwf_destroy_plan(fft_plan_forward[i][j]);
            fftwf_destroy_plan(fft_plan_inverse[i][j]);
        }
    }
    for (int i=0; i<depth; ++i) {
        for (int j=0; j<=1; ++j) {
            fftwf_free(pool[i][j]);
            fftwf_free(pool_tmp[i][j]);
            fftwf_free(pool_tmp_pv[i][j]);
        }
    }
    return output;
}