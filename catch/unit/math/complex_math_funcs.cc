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
#include "binary_common.hh"

COMPLEX_BINARY_KERNEL_DEF(hipCadd)

template <typename T> T hipCadd_ref(T x1, T x2) {
  T y;
  y.x = x1.x + x2.x;
  y.y = x1.y + x2.y;
  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCadd_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexBinaryFloatingPointSpecialValuesTest(
      hipCadd_kernel<TestType>, hipCadd_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(0)));
}

COMPLEX_BINARY_KERNEL_DEF(hipCsub)

template <typename T> T hipCsub_ref(T x1, T x2) {
  T y;
  y.x = x1.x - x2.x;
  y.y = x1.y - x2.y;
  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCsub_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexBinaryFloatingPointSpecialValuesTest(
      hipCsub_kernel<TestType>, hipCsub_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(0)));
}

COMPLEX_BINARY_KERNEL_DEF(hipCmul)

template <typename T> T hipCmul_ref(T x1, T x2) {
  T y;
  y.x = x1.x * x2.x - x1.y * x2.y;
  y.y = x1.y * x2.x + x1.x * x2.y;
  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCmul_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexBinaryFloatingPointSpecialValuesTest(
      hipCmul_kernel<TestType>, hipCmul_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(1)));
}

COMPLEX_BINARY_KERNEL_DEF(hipCdiv)

template <typename T> T hipCdiv_ref(T x1, T x2) {
  decltype(T().x) sqabs = x2.x * x2.x + x2.y * x2.y;
  T y;
  y.x = (x1.x * x2.x + x1.y * x2.y) / sqabs;
  y.y = (x1.y * x2.x - x1.x * x2.y) / sqabs;
  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCdiv_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexBinaryFloatingPointSpecialValuesTest(
      hipCdiv_kernel<TestType>, hipCdiv_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(2)));
}

COMPLEX_TERNARY_KERNEL_DEF(hipCfma)

template <typename T> T hipCfma_ref(T x1, T x2, T x3) {
  T y;
  decltype(T().x) real = (x1.x * x2.x) + x3.x;
  decltype(T().x) imag = (x2.x * x1.y) + x3.y;

  y.x = -(x1.y * x2.y) + real;
  y.y = (x1.x * x2.y) + imag; 

  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCfma_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexTernaryFloatingPointSpecialValuesTest(
      hipCfma_kernel<TestType>, hipCfma_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(2)));
}

COMPLEX_UNARY_KERNEL_DEF(hipConj)

template <typename T> T hipConj_ref(T x1) {
  T y;
  y.x = x1.x;
  y.y = -x1.y;
  return y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipConj_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexUnaryFloatingPointSpecialValuesTest(
      hipConj_kernel<TestType, TestType>, hipConj_ref<TestType>,
      ComplexValidatorBuilderFactory<TestType>(ULPValidatorBuilderFactory<decltype(TestType().x)>(0)));
}

COMPLEX_UNARY_KERNEL_DEF(hipCsqabs)

template <typename T> decltype(T().x) hipCsqabs_ref(T x) {
  return x.x * x.x + x.y * x.y;
}

TEMPLATE_TEST_CASE("Unit_Device_hipCsqabs_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexUnaryFloatingPointSpecialValuesTest(
      hipCsqabs_kernel<decltype(TestType().x), TestType>, hipCsqabs_ref<TestType>,
      ULPValidatorBuilderFactory<decltype(TestType().x)>(1));
}

COMPLEX_UNARY_KERNEL_DEF(hipCabs)

template <typename T> decltype(T().x) hipCabs_ref(T x) {
  return std::sqrt(x.x * x.x + x.y * x.y);
}

TEMPLATE_TEST_CASE("Unit_Device_hipCabs_Accuracy_Positive", "", hipFloatComplex, hipDoubleComplex) {
  ComplexUnaryFloatingPointSpecialValuesTest(
      hipCabs_kernel<decltype(TestType().x), TestType>, hipCabs_ref<TestType>,
      ULPValidatorBuilderFactory<decltype(TestType().x)>(1));
}
