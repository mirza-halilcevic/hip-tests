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

#include "unary_common.hh"
#include <math.h>


#define UNARY_TRIG_TEST_DEF(func_name, sp_ulp, dp_ulp)                                             \
  MATH_UNARY_KERNEL_DEF(func_name)                                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #func_name "f_Accuracy_Positive") {                                     \
    UnarySinglePrecisionTest(func_name##_kernel<float>, std::func_name,                            \
                             ULPValidatorBuilderFactory<float>(sp_ulp));                           \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #func_name "_Accuracy_Positive") {                                      \
    UnaryDoublePrecisionTest(func_name##_kernel<double>, std::func_name,                           \
                             ULPValidatorBuilderFactory<double>(dp_ulp));                          \
  }

UNARY_TRIG_TEST_DEF(sin, 2, 2)

UNARY_TRIG_TEST_DEF(cos, 2, 2)

UNARY_TRIG_TEST_DEF(tan, 4, 2)

UNARY_TRIG_TEST_DEF(asin, 2, 2)

UNARY_TRIG_TEST_DEF(acos, 2, 2)

UNARY_TRIG_TEST_DEF(atan, 2, 2)

UNARY_TRIG_TEST_DEF(sinh, 3, 2)

UNARY_TRIG_TEST_DEF(cosh, 2, 1)

UNARY_TRIG_TEST_DEF(tanh, 2, 1)

UNARY_TRIG_TEST_DEF(asinh, 3, 2)

UNARY_TRIG_TEST_DEF(acosh, 4, 2)

UNARY_TRIG_TEST_DEF(atanh, 3, 2)
