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

#include "casting_common.hh"

#define CAST_FLOAT2INT_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RINT_REF_DEF(kern_name, T, float, round_dir)                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref,                                         \
                             EqValidatorBuilderFactory<T>(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());                                      \
  }                                                                                                \

CAST_FLOAT2INT_TEST_DEF(int, float2int_rd, FE_DOWNWARD)
CAST_FLOAT2INT_TEST_DEF(int, float2int_rn, FE_TONEAREST)
CAST_FLOAT2INT_TEST_DEF(int, float2int_ru, FE_UPWARD)
CAST_FLOAT2INT_TEST_DEF(int, float2int_rz, FE_TOWARDZERO)

CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_rd, FE_DOWNWARD)
CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_rn, FE_TONEAREST)
CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_ru, FE_UPWARD)
CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_rz, FE_TOWARDZERO)

#define CAST_FLOAT2LL_TEST_DEF(T, kern_name, round_dir)                                            \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RINT_REF_DEF(kern_name, T, float, round_dir)                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref,                               \
                             EqValidatorBuilderFactory<T>(), static_cast<float>(std::numeric_limits<T>::min()), static_cast<float>(std::numeric_limits<T>::max()));                              \
  }

CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_rd, FE_DOWNWARD)
CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_rn, FE_TONEAREST)
CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_ru, FE_UPWARD)
CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_rz, FE_TOWARDZERO)

CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_rd, FE_DOWNWARD)
CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_rn, FE_TONEAREST)
CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_ru, FE_UPWARD)
CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_rz, FE_TOWARDZERO)

CAST_KERNEL_DEF(float_as_int, int, float)

TEST_CASE("Unit_Device_float_as_int_Positive") {
  int (*ref)(float) = type2_as_type1_ref<int, float>;
  UnarySinglePrecisionTest(float_as_int_kernel, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(float_as_uint, unsigned int, float)

TEST_CASE("Unit_Device_float_as_uint_Positive") {
  unsigned int (*ref)(float) = type2_as_type1_ref<unsigned int, float>;
  UnarySinglePrecisionTest(float_as_uint_kernel, ref, EqValidatorBuilderFactory<unsigned int>());
}
