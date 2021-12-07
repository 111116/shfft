Fast and Accurate Spherical Harmonics Triple and Multiple Products

precomputed files

- `sparseN`: tripling tensor for regular product of order-N SH vectors
- `sh2fs.cpp`, `fs2sh.cpp`: SH - Fourier series conversion coefficients

其中`fs2sh.cpp` 由于利用了由SH转换或相乘得到的傅里叶级数系数的对称性，不适用于将一般的傅里叶级数投影到SH。

### 编译运行

依赖库（各分支以`Makefile`为准）：Intel MKL

修改 `shorder.hpp` 中的SH阶数，然后执行

```bash
make && ./test
```

### 数据性质

SH向量转换为复形式傅里叶级数后系数满足对称性：
```c++
if j%2==0
	a.at(i,j) = a.at(-i,j)
	a.at(i,j) = conj(a.at(i,-j))
else
	a.at(i,j) = -a.at(-i,j)
	a.at(i,j) = -conj(a.at(i,-j))
```

### 算法说明

以4阶（$0\le l\le 3$）为例

#### 第一步 `sh2fs.cpp`

利用预处理的系数，把4阶SH向量转换为傅里叶级数形式（下标绝对值不超过3）
$$
\sum_{a=-3}^3 \sum_{b=-3}^3 f_{a,b} e^{i(a\theta+b\phi)}
$$
$f$具有$7\times7$个复系数

#### 第二步 `fastmul.cpp`

把两个傅里叶级数相乘，结果为下标绝对值不超过6的傅里叶级数，做法如下

1. 对输入的两个傅里叶级数的系数矩阵，补零，作16×16的DFT。
2. 把两个16×16的结果矩阵按元素相乘
3. 作IDFT，得出结果的傅里叶级数的系数矩阵

第1步中可以跳过输入全0的行的一维子变换；第3步中可以跳过输出不被使用的列的一维子变换。

过程中数值需要乘上系数$1/256$。结果的复系数个数是13×13。

#### 第三步 `fs2sh.cpp`

利用预处理的系数，把下标绝对值不超过6的傅里叶级数转换为4阶SH向量。

由于数据具有对称性，实际只使用了傅里叶级数中两维下标均不大于0的系数的值。

### 备注

一个理论可行的算法优化是：利用傅里叶级数系数的共轭对称性质 $a_{i,j} = \overline{a_{-i,-j}}$ ，使用实数DFT，可节省约一半运算量。但实测 Intel IPP 与 Intel MKL 小数据实数DFT比一般复数DFT更慢，FFTW实数DFT比一般复数DFT稍快一些。只进行DFT执行过程的耗时如下

```plain
FFTW DFT executions (n is order of SH) time / 10ns

original DFT
n=8: 530 500 535
n=7: 503 507 528
n=6: 512 509 518
n=5: 181 188 185

real-valued DFT
n=8: 469 464 495
n=7: 434 436 461
n=6: 432 433 428
n=5: 164 177 174
```

