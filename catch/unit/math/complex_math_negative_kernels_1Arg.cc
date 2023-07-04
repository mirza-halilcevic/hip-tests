/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hip_complex.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_SHELL_ONE_ARG_FLOAT(func_name)                                                      \
  __global__ void func_name##_kernel_v1(float* result, hipFloatComplex* x) { *result = func_name(x); } \
  __global__ void func_name##_kernel_v2(float* result, hipDoubleComplex x) { *result = func_name(x); } \
  __global__ void func_name##_kernel_v3(float* result, float x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v4(float* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v5(hipFloatComplex* result, hipFloatComplex x) { *result = func_name(x); }              \
  __global__ void func_name##_kernel_v6(Dummy* result, hipFloatComplex x) { *result = func_name(x); }              \
  void func_name##_v1(float* result, hipFloatComplex* x) { *result = func_name(x); }                               \
  void func_name##_v2(float* result, hipDoubleComplex x) { *result = func_name(x); }                               \
  void func_name##_v3(float* result, float x) { *result = func_name(x); }                                \
  void func_name##_v4(float* result, Dummy x) { *result = func_name(x); }                             \
  void func_name##_v5(hipFloatComplex* result, hipFloatComplex x) { *result = func_name(x); }                              \
  void func_name##_v6(Dummy* result, hipFloatComplex x) { *result = func_name(x); }

#define NEGATIVE_SHELL_ONE_ARG_DOUBLE(func_name)                                                      \
  __global__ void func_name##_kernel_v1(double* result, hipDoubleComplex* x) { *result = func_name(x); } \
  __global__ void func_name##_kernel_v2(double* result, hipFloatComplex x) { *result = func_name(x); } \
  __global__ void func_name##_kernel_v3(double* result, double x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v4(double* result, Dummy x) { *result = func_name(x); }            \
  __global__ void func_name##_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = func_name(x); }              \
  __global__ void func_name##_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = func_name(x); }              \
  void func_name##_v1(double* result, hipDoubleComplex* x) { *result = func_name(x); }                               \
  void func_name##_v2(double* result, hipFloatComplex x) { *result = func_name(x); }                               \
  void func_name##_v3(double* result, double x) { *result = func_name(x); }                                \
  void func_name##_v4(double* result, Dummy x) { *result = func_name(x); }                             \
  void func_name##_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = func_name(x); }                              \
  void func_name##_v6(Dummy* result, hipDoubleComplex x) { *result = func_name(x); }

__global__ void hipConjf_kernel_v1(hipFloatComplex* result, hipFloatComplex* x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v2(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v6(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConjf(x); }
__global__ void hipConjf_kernel_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v1(hipFloatComplex* result, hipFloatComplex* x) { *result = hipConjf(x); }
void hipConjf_v2(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConjf(x); }
void hipConjf_v3(hipFloatComplex* result, float x) { *result = hipConjf(x); }
void hipConjf_v4(hipFloatComplex* result, Dummy x) { *result = hipConjf(x); }
void hipConjf_v5(float* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v6(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConjf(x); }
void hipConjf_v7(Dummy* result, hipFloatComplex x) { *result = hipConjf(x); }

__global__ void hipConj_kernel_v1(hipDoubleComplex* result, hipDoubleComplex* x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v2(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v6(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConj(x); }
__global__ void hipConj_kernel_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v1(hipDoubleComplex* result, hipDoubleComplex* x) { *result = hipConj(x); }
void hipConj_v2(hipDoubleComplex* result, hipFloatComplex x) { *result = hipConj(x); }
void hipConj_v3(hipDoubleComplex* result, double x) { *result = hipConj(x); }
void hipConj_v4(hipDoubleComplex* result, Dummy x) { *result = hipConj(x); }
void hipConj_v5(double* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v6(hipFloatComplex* result, hipDoubleComplex x) { *result = hipConj(x); }
void hipConj_v7(Dummy* result, hipDoubleComplex x) { *result = hipConj(x); }

NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCsqabsf)
NEGATIVE_SHELL_ONE_ARG_FLOAT(hipCabsf)
NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCsqabs)
NEGATIVE_SHELL_ONE_ARG_DOUBLE(hipCabs)