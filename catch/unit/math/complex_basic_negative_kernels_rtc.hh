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

#pragma once

static constexpr auto kReal{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCrealf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v3(float* result, float x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v4(float* result, Dummy x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCrealf(x); }
  __global__ void hipCrealf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCrealf(x); }
  __global__ void hipCreal_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v2(double* result, hipFloatComplex x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v3(double* result, double x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v4(double* result, Dummy x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCreal(x); }
  __global__ void hipCreal_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCreal(x); }
  void hipCrealf_v1(float* result, hipFloatComplex* x) { *result = hipCrealf(x); }
  void hipCrealf_v2(float* result, hipDoubleComplex x) { *result = hipCrealf(x); }
  void hipCrealf_v3(float* result, float x) { *result = hipCrealf(x); }
  void hipCrealf_v4(float* result, Dummy x) { *result = hipCrealf(x); }
  void hipCrealf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCrealf(x); }
  void hipCrealf_v6(Dummy* result, hipFloatComplex x) { *result = hipCrealf(x); }
  void hipCreal_v1(double* result, hipDoubleComplex* x) { *result = hipCreal(x); }
  void hipCreal_v2(double* result, hipFloatComplex x) { *result = hipCreal(x); }
  void hipCreal_v3(double* result, double x) { *result = hipCreal(x); }
  void hipCreal_v4(double* result, Dummy x) { *result = hipCreal(x); }
  void hipCreal_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCreal(x); }
  void hipCreal_v6(Dummy* result, hipDoubleComplex x) { *result = hipCreal(x); }
)"};

static constexpr auto kImag{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipCimagf_kernel_v1(float* result, hipFloatComplex* x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v2(float* result, hipDoubleComplex x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v3(float* result, float x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v4(float* result, Dummy x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCimagf(x); }
  __global__ void hipCimagf_kernel_v6(Dummy* result, hipFloatComplex x) { *result = hipCimagf(x); }
  __global__ void hipCimag_kernel_v1(double* result, hipDoubleComplex* x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v2(double* result, hipFloatComplex x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v3(double* result, double x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v4(double* result, Dummy x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCimag(x); }
  __global__ void hipCimag_kernel_v6(Dummy* result, hipDoubleComplex x) { *result = hipCimag(x); }
  void hipCimagf_v1(float* result, hipFloatComplex* x) { *result = hipCimagf(x); }
  void hipCimagf_v2(float* result, hipDoubleComplex x) { *result = hipCimagf(x); }
  void hipCimagf_v3(float* result, float x) { *result = hipCimagf(x); }
  void hipCimagf_v4(float* result, Dummy x) { *result = hipCimagf(x); }
  void hipCimagf_v5(hipFloatComplex* result, hipFloatComplex x) { *result = hipCimagf(x); }
  void hipCimagf_v6(Dummy* result, hipFloatComplex x) { *result = hipCimagf(x); }
  void hipCimag_v1(double* result, hipDoubleComplex* x) { *result = hipCimag(x); }
  void hipCimag_v2(double* result, hipFloatComplex x) { *result = hipCimag(x); }
  void hipCimag_v3(double* result, double x) { *result = hipCimag(x); }
  void hipCimag_v4(double* result, Dummy x) { *result = hipCimag(x); }
  void hipCimag_v5(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipCimag(x); }
  void hipCimaghipCreal_v6(Dummy* result, hipDoubleComplex x) { *result = hipCimag(x); }
)"};

static constexpr auto kComplexDoubleToFloat{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipComplexDoubleToFloat_kernel_v1(hipFloatComplex* result, hipDoubleComplex* x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v2(hipFloatComplex* result, hipFloatComplex x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v3(hipFloatComplex* result, double x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v4(hipFloatComplex* result, Dummy x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v5(float* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v6(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
  __global__ void hipComplexDoubleToFloat_kernel_v7(Dummy* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v1(hipFloatComplex* result, hipDoubleComplex* x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v2(hipFloatComplex* result, hipFloatComplex x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v3(hipFloatComplex* result, double x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v4(hipFloatComplex* result, Dummy x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v5(float* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v6(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
  void hipComplexDoubleToFloat_v7(Dummy* result, hipDoubleComplex x) { *result = hipComplexDoubleToFloat(x); }
)"};

static constexpr auto kComplexFloatToDouble{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hipComplexFloatToDouble_kernel_v1(hipDoubleComplex* result, hipFloatComplex* x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v2(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v3(hipDoubleComplex* result, float x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v4(hipDoubleComplex* result, Dummy x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v5(double* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v6(hipFloatComplex* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
  __global__ void hipComplexFloatToDouble_kernel_v7(Dummy* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v1(hipDoubleComplex* result, hipFloatComplex* x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v2(hipDoubleComplex* result, hipDoubleComplex x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v3(hipDoubleComplex* result, float x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v4(hipDoubleComplex* result, Dummy x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v5(double* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v6(hipFloatComplex* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
  void hipComplexFloatToDouble_v7(Dummy* result, hipFloatComplex x) { *result = hipComplexFloatToDouble(x); }
)"};


