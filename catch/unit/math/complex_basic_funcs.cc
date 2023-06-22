/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "complex_common.hh"

template <typename T>
__device__ __host__ void complex_make_func(T* const ys, decltype(T().x) const x1s,
                                  decltype(T().x) const x2s) {

  if constexpr (std::is_same_v<hipFloatComplex, T>) {
    *ys = make_hipFloatComplex(x1s, x2s);
  } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
    *ys = make_hipDoubleComplex(x1s, x2s);
  }
}

template <typename T>
__global__ void complex_make_kernel(T* const ys, decltype(T().x) const x1s,
                                  decltype(T().x) const x2s) {
  complex_make_func(ys, x1s, x2s);
}

TEMPLATE_TEST_CASE("Unit_Device_make_hipComplex_Device_Positive", "", hipFloatComplex, hipDoubleComplex) {
  decltype(TestType().x) x1 = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) x2 = GENERATE(-1.75, 0, 1.75);

  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  complex_make_kernel<TestType><<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0].x == x1);
  REQUIRE(y.ptr()[0].y == x2);
}

TEMPLATE_TEST_CASE("Unit_Device_make_hipComplex_Host_Positive", "", hipFloatComplex, hipDoubleComplex) {
  float x1 = GENERATE(-0.25, 0, 0.25);
  float x2 = GENERATE(-1.75, 0, 1.75);
  TestType y;

  complex_make_func(&y, x1, x2);

  REQUIRE(y.x == x1);
  REQUIRE(y.y == x2);
}

template <typename T>
__device__ __host__ void complex_real_imag_func(decltype(T().x)* const y1s, decltype(T().x)* const y2s,
                                                T const xs) {

  if constexpr (std::is_same_v<hipFloatComplex, T>) {
    *y1s = hipCrealf(xs);
    *y2s = hipCimagf(xs);
  } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {
    *y1s = hipCreal(xs);
    *y2s = hipCimag(xs);
  }
}

template <typename T>
__global__ void complex_real_imag_kernel(decltype(T().x)* const y1s, decltype(T().x)* const y2s,
                                                T const xs) {
  complex_real_imag_func(y1s, y2s, xs);
}

TEMPLATE_TEST_CASE("Unit_Device_hipCreal_hipCimag_Device_Positive", "", hipFloatComplex, hipDoubleComplex) {
  decltype(TestType().x) x1 = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) x2 = GENERATE(-1.75, 0, 1.75);
  TestType x;
  complex_make_func(&x, x1, x2);

  LinearAllocGuard<decltype(TestType().x)> real(LinearAllocs::hipMallocManaged, sizeof(decltype(TestType().x)));
  LinearAllocGuard<decltype(TestType().x)> imag(LinearAllocs::hipMallocManaged, sizeof(decltype(TestType().x)));

  complex_real_imag_kernel<TestType><<<1, 1>>>(real.ptr(), imag.ptr(), x);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(real.ptr()[0] == x1);
  REQUIRE(imag.ptr()[0] == x2);
}

TEMPLATE_TEST_CASE("Unit_Device_hipCreal_hipCimag_Host_Positive", "", hipFloatComplex, hipDoubleComplex) {
  float x1 = GENERATE(-0.25, 0, 0.25);
  float x2 = GENERATE(-1.75, 0, 1.75);
  TestType x;
  complex_make_func(&x, x1, x2);

  decltype(TestType().x) real, imag;
  complex_real_imag_func(&real, &imag, x);

  REQUIRE(real == x1);
  REQUIRE(imag == x2);
}

template <typename T1, typename T2>
__device__ __host__ void complex_cast_func(T1* const ys, T2 const xs) {

  if constexpr (std::is_same_v<hipFloatComplex, T1>) {
    *ys = hipComplexDoubleToFloat(xs);
  } else if constexpr (std::is_same_v<hipDoubleComplex, T1>) {
    *ys = hipComplexFloatToDouble(xs);
  }
}

template <typename T1, typename T2>
__global__ void complex_cast_kernel(T1* const ys, T2 const xs) {
  complex_cast_func(ys, xs);
}

TEST_CASE("Unit_Device_hipComplexDoubleToFloat_Device_Positive") {
  double x1 = GENERATE(-0.25, 0, 0.25);
  double x2 = GENERATE(-1.75, 0, 1.75);
  hipDoubleComplex x;
  complex_make_func(&x, x1, x2);

  LinearAllocGuard<hipFloatComplex> y(LinearAllocs::hipMallocManaged, sizeof(hipFloatComplex));

  complex_cast_kernel<<<1, 1>>>(y.ptr(), x);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0].x == static_cast<float>(x1));
  REQUIRE(y.ptr()[0].y == static_cast<float>(x2));
}

TEST_CASE("Unit_Device_hipComplexDoubleToFloat_Host_Positive") {
  double x1 = GENERATE(-0.25, 0, 0.25);
  double x2 = GENERATE(-1.75, 0, 1.75);
  hipDoubleComplex x;
  complex_make_func(&x, x1, x2);

  hipFloatComplex y;
  complex_cast_func(&y, x);

  REQUIRE(y.x == static_cast<float>(x1));
  REQUIRE(y.y == static_cast<float>(x2));
}

TEST_CASE("Unit_Device_hipComplexFloatToDouble_Device_Positive") {
  float x1 = GENERATE(-0.25, 0, 0.25);
  float x2 = GENERATE(-1.75, 0, 1.75);
  hipFloatComplex x;
  complex_make_func(&x, x1, x2);

  LinearAllocGuard<hipDoubleComplex> y(LinearAllocs::hipMallocManaged, sizeof(hipDoubleComplex));

  complex_cast_kernel<<<1, 1>>>(y.ptr(), x);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0].x == static_cast<double>(x1));
  REQUIRE(y.ptr()[0].y == static_cast<double>(x2));
}

TEST_CASE("Unit_Device_hipComplexFloatToDouble_Host_Positive") {
  float x1 = GENERATE(-0.25, 0, 0.25);
  float x2 = GENERATE(-1.75, 0, 1.75);
  hipFloatComplex x;
  complex_make_func(&x, x1, x2);

  hipDoubleComplex y;
  complex_cast_func(&y, x);

  REQUIRE(y.x == static_cast<double>(x1));
  REQUIRE(y.y == static_cast<double>(x2));
}