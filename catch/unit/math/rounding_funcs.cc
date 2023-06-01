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
#include "binary_common.hh"
#include "ternary_common.hh"
#include "quaternary_common.hh"
#include <fenv.h>


#define CAST_KERNEL_DEF(func_name)                                                                 \
  template <typename T1, typename T2>                                                              \
  __global__ void func_name##_kernel(T1* const ys, const size_t num_xs, T2* const xs) {            \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
        ys[i] = __##func_name(xs[i]);                                                              \
    }                                                                                              \
  }

/*
CAST_KERNEL_DEF(double2int_ru)

TEST_CASE("Unit_double2int_ru_Positive") {
  auto double2int_ref = [](double arg) -> int {
    if (arg >= static_cast<double>(std::numeric_limits<int>::max()))
      arg = static_cast<double>(std::numeric_limits<int>::max());
    else if (arg <= static_cast<double>(std::numeric_limits<int>::min()))
      arg = static_cast<double>(std::numeric_limits<int>::min());
    return std::ceil(arg); };
  int (*ref)(double) = double2int_ref;
  UnaryDoublePrecisionTest(double2int_ru_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double2int_rd)

TEST_CASE("Unit_double2int_rd_Positive") {
  auto double2int_ref = [](double arg) -> int {
    if (arg >= static_cast<double>(std::numeric_limits<int>::max()))
      arg = static_cast<double>(std::numeric_limits<int>::max());
    else if (arg <= static_cast<double>(std::numeric_limits<int>::min()))
      arg = static_cast<double>(std::numeric_limits<int>::min());
    return std::floor(arg); };
  int (*ref)(double) = double2int_ref;
  UnaryDoublePrecisionTest(double2int_rd_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double2int_rn)

TEST_CASE("Unit_double2int_rn_Positive") {
  auto double2int_ref = [](double arg) -> int {
    if (arg >= static_cast<double>(std::numeric_limits<int>::max()))
      arg = static_cast<double>(std::numeric_limits<int>::max());
    else if (arg <= static_cast<double>(std::numeric_limits<int>::min()))
      arg = static_cast<double>(std::numeric_limits<int>::min());
    return std::rint(arg); };
  int (*ref)(double) = double2int_ref;
  UnaryDoublePrecisionTest(double2int_rn_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double2int_rz)

TEST_CASE("Unit_double2int_rz_Positive") {
  auto double2int_ref = [](double arg) -> int {
    if (arg >= static_cast<double>(std::numeric_limits<int>::max()))
      return static_cast<double>(std::numeric_limits<int>::max());
    else if (arg <= static_cast<double>(std::numeric_limits<int>::min()))
      return std::numeric_limits<int>::min();
    return static_cast<int>(arg); };
  int (*ref)(double) = double2int_ref;
  UnaryDoublePrecisionTest(double2int_rz_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double2uint_ru)

TEST_CASE("Unit_double2uint_ru_Positive") {
  auto double2uint_ref = [](double arg) -> unsigned int {
    if (arg >= static_cast<double>(std::numeric_limits<unsigned int>::max()))
      return std::numeric_limits<unsigned int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<unsigned int>::min()))
      return std::numeric_limits<unsigned int>::min();
    return std::ceil(arg); };
  unsigned int (*ref)(double) = double2uint_ref;
  UnaryDoublePrecisionTest(double2uint_ru_kernel<unsigned int, double>, ref, EqValidatorBuilderFactory<unsigned int>());
}

CAST_KERNEL_DEF(double2uint_rd)

TEST_CASE("Unit_double2uint_rd_Positive") {
  auto double2uint_ref = [](double arg) -> unsigned int {
    if (arg >= static_cast<double>(std::numeric_limits<unsigned int>::max()))
      return std::numeric_limits<unsigned int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<unsigned int>::min()))
      return std::numeric_limits<unsigned int>::min();
    return std::floor(arg); };
  unsigned int (*ref)(double) = double2uint_ref;
  UnaryDoublePrecisionTest(double2uint_rd_kernel<unsigned int, double>, ref, EqValidatorBuilderFactory<unsigned int>());
}

CAST_KERNEL_DEF(double2uint_rn)

TEST_CASE("Unit_double2uint_rn_Positive") {
  auto double2uint_ref = [](double arg) -> unsigned int {
    if (arg >= static_cast<double>(std::numeric_limits<unsigned int>::max()))
      return std::numeric_limits<unsigned int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<unsigned int>::min()))
      return std::numeric_limits<unsigned int>::min();
    return std::rint(arg); };
  unsigned int (*ref)(double) = double2uint_ref;
  UnaryDoublePrecisionTest(double2uint_rn_kernel<unsigned int, double>, ref, EqValidatorBuilderFactory<unsigned int>());
}

CAST_KERNEL_DEF(double2uint_rz)

TEST_CASE("Unit_double2uint_rz_Positive") {
  auto double2uint_ref = [](double arg) -> unsigned int {
    if (arg >= static_cast<double>(std::numeric_limits<unsigned int>::max()))
      return std::numeric_limits<unsigned int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<unsigned int>::min()))
      return std::numeric_limits<unsigned int>::min();
    return static_cast<unsigned int>(arg); };
  unsigned int (*ref)(double) = double2uint_ref;
  UnaryDoublePrecisionTest(double2uint_rz_kernel<unsigned int, double>, ref, EqValidatorBuilderFactory<unsigned int>());
}

CAST_KERNEL_DEF(double2ll_ru)

TEST_CASE("Unit_double2ll_ru_Positive") {
  auto double2ll_ref = [](double arg) -> long long int {
    if (arg >= static_cast<double>(std::numeric_limits<long long int>::max()))
      return std::numeric_limits<long long int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<long long int>::min()))
      return std::numeric_limits<long long int>::min();
    return std::ceil(arg); };
  long long int (*ref)(double) = double2ll_ref;
  UnaryDoublePrecisionBruteForceTest(double2ll_ru_kernel<long long int, double>, ref, EqValidatorBuilderFactory<long long int>(), static_cast<double>(std::numeric_limits<long long int>::min()), static_cast<double>(std::numeric_limits<long long int>::max()));
}

CAST_KERNEL_DEF(double2ll_rd)

TEST_CASE("Unit_double2ll_rd_Positive") {
  auto double2ll_ref = [](double arg) -> long long int {
    if (arg >= static_cast<double>(std::numeric_limits<long long int>::max()))
      return std::numeric_limits<long long int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<long long int>::min()))
      return std::numeric_limits<long long int>::min();
    return std::floor(arg); };
  long long int (*ref)(double) = double2ll_ref;
  UnaryDoublePrecisionBruteForceTest(double2ll_rd_kernel<long long int, double>, ref, EqValidatorBuilderFactory<long long int>(), static_cast<double>(std::numeric_limits<long long int>::min()), static_cast<double>(std::numeric_limits<long long int>::max()));
}

CAST_KERNEL_DEF(double2ll_rn)

TEST_CASE("Unit_double2ll_rn_Positive") {
  auto double2ll_ref = [](double arg) -> long long int {
    if (arg >= static_cast<double>(std::numeric_limits<long long int>::max()))
      return std::numeric_limits<long long int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<long long int>::min()))
      return std::numeric_limits<long long int>::min();
    return std::rint(arg); };
  long long int (*ref)(double) = double2ll_ref;
  UnaryDoublePrecisionBruteForceTest(double2ll_rn_kernel<long long int, double>, ref, EqValidatorBuilderFactory<long long int>(), static_cast<double>(std::numeric_limits<long long int>::min()), static_cast<double>(std::numeric_limits<long long int>::max()));
}

CAST_KERNEL_DEF(double2ll_rz)

TEST_CASE("Unit_double2ll_rz_Positive") {
  auto double2ll_ref = [](double arg) -> long long int {
    if (arg >= static_cast<double>(std::numeric_limits<long long int>::max()))
      return std::numeric_limits<long long int>::max();
    else if (arg <= static_cast<double>(std::numeric_limits<long long int>::min()))
      return std::numeric_limits<long long int>::min();
    return static_cast<long long int>(arg); };
  long long int (*ref)(double) = double2ll_ref;
  UnaryDoublePrecisionBruteForceTest(double2ll_rz_kernel<long long int, double>, ref, EqValidatorBuilderFactory<long long int>(), static_cast<double>(std::numeric_limits<long long int>::min()), static_cast<double>(std::numeric_limits<long long int>::max()));
}*/

#define CAST_DOUBLE2INT_TEST_DEF(T, kern_name, round_dir)                                          \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  T kern_name##_ref(double arg) {                                                                  \
    int curr_direction = fegetround();                                                             \
    if (arg >= static_cast<double>(std::numeric_limits<T>::max()))                                 \
      return std::numeric_limits<T>::max();                                                        \
    else if (arg <= static_cast<double>(std::numeric_limits<T>::min()))                            \
      return std::numeric_limits<T>::min();                                                        \
    fesetround(round_dir);                                                                         \
    double result = std::rint(arg);                                                                \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionTest(kern_name##_kernel<T, double>, ref,                                   \
                             EqValidatorBuilderFactory<T>());                                      \
  }                                                                                                \

CAST_DOUBLE2INT_TEST_DEF(int, double2int_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(int, double2int_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(int, double2int_ru, FE_UPWARD)
CAST_DOUBLE2INT_TEST_DEF(int, double2int_rz, FE_TOWARDZERO)

CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_ru, FE_UPWARD)
CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_rz, FE_TOWARDZERO)

#define CAST_DOUBLE2LL_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  T kern_name##_ref(double arg) {                                                                  \
    int curr_direction = fegetround();                                                             \
    if (arg >= static_cast<double>(std::numeric_limits<T>::max()))                                 \
      return std::numeric_limits<T>::max();                                                        \
    else if (arg <= static_cast<double>(std::numeric_limits<T>::min()))                            \
      return std::numeric_limits<T>::min();                                                        \
    fesetround(round_dir);                                                                         \
    double result = rint(arg);                                                                     \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel<T, double>, ref,                         \
                             EqValidatorBuilderFactory<T>(), static_cast<double>(std::numeric_limits<T>::min()), static_cast<double>(std::numeric_limits<T>::max()));                              \
  }

CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_rd, FE_DOWNWARD)
CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_rn, FE_TONEAREST)
CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_ru, FE_UPWARD)
CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_rz, FE_TOWARDZERO)

CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_rd, FE_DOWNWARD)
CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_rn, FE_TONEAREST)
CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_ru, FE_UPWARD)
CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_rz, FE_TOWARDZERO)

#define CAST_DOUBLE2FLOAT_TEST_DEF(kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  float kern_name##_ref(double arg) {                                                              \
    int curr_direction = fegetround();                                                             \
    fesetround(round_dir);                                                                         \
    float result = static_cast<float>(arg);                                                        \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    UnaryDoublePrecisionTest(kern_name##_kernel<float, double>, ref,                               \
                             EqValidatorBuilderFactory<float>());                                  \
  }                                                                                                \

CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rd, FE_DOWNWARD)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rn, FE_TONEAREST)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_ru, FE_UPWARD)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rz, FE_TOWARDZERO)

CAST_KERNEL_DEF(double2hiint)

int double2hiint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp)); 
  return tmp[1];
}

TEST_CASE("Unit_Device_double2hiint_Positive") {
  int (*ref)(double) = double2hiint_ref;
  UnaryDoublePrecisionTest(double2hiint_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double2loint)

int double2loint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp)); 
  return tmp[0];
}

TEST_CASE("Unit_Device_double2loint_Positive") {
  int (*ref)(double) = double2loint_ref;
  UnaryDoublePrecisionTest(double2loint_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(double_as_longlong)

long long int double_as_longlong_ref(double arg) {
  long long int tmp;
  memcpy(&tmp, &arg, sizeof(tmp)); 
  return tmp;
}

TEST_CASE("Unit_Device_double_as_longlong_Positive") {
  long long int (*ref)(double) = double_as_longlong_ref;
  UnaryDoublePrecisionTest(double_as_longlong_kernel<long long int, double>, ref, EqValidatorBuilderFactory<long long int>());
}

#define CAST_FLOAT2INT_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  T kern_name##_ref(float arg) {                                                                   \
    int curr_direction = fegetround();                                                             \
    if (arg >= static_cast<float>(std::numeric_limits<T>::max()))                                  \
      return std::numeric_limits<T>::max();                                                        \
    else if (arg <= static_cast<float>(std::numeric_limits<T>::min()))                             \
      return std::numeric_limits<T>::min();                                                        \
    fesetround(round_dir);                                                                         \
    float result = rint(arg);                                                                      \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel<T, float>, ref,                               \
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
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  T kern_name##_ref(float arg) {                                                                   \
    int curr_direction = fegetround();                                                             \
    if (arg >= static_cast<float>(std::numeric_limits<T>::max()))                                  \
      return std::numeric_limits<T>::max();                                                        \
    else if (arg <= static_cast<float>(std::numeric_limits<T>::min()))                             \
      return std::numeric_limits<T>::min();                                                        \
    fesetround(round_dir);                                                                         \
    float result = rint(arg);                                                                      \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel<T, float>, ref,                               \
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

CAST_KERNEL_DEF(float_as_int)

int float_as_int_ref(float arg) {
  int tmp;
  memcpy(&tmp, &arg, sizeof(tmp)); 
  return tmp;
}

TEST_CASE("Unit_Device_float_as_int_Positive") {
  int (*ref)(float) = float_as_int_ref;
  UnarySinglePrecisionTest(float_as_int_kernel<int, float>, ref, EqValidatorBuilderFactory<int>());
}

CAST_KERNEL_DEF(float_as_uint)

unsigned int float_as_uint_ref(float arg) {
  unsigned int tmp;
  memcpy(&tmp, &arg, sizeof(tmp)); 
  return tmp;
}

TEST_CASE("Unit_Device_float_as_uint_Positive") {
  unsigned int (*ref)(float) = float_as_uint_ref;
  UnarySinglePrecisionTest(float_as_uint_kernel<unsigned int, float>, ref, EqValidatorBuilderFactory<unsigned int>());
}

#define CAST_INT2FLOAT_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  float kern_name##_ref(T arg) {                                                                   \
    int curr_direction = fegetround();                                                             \
    fesetround(round_dir);                                                                         \
    float result = static_cast<float>(arg);                                                        \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(T) = kern_name##_ref;                                                           \
    UnaryIntRangeTest(kern_name##_kernel<float, T>, ref,                                         \
                             EqValidatorBuilderFactory<float>(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());  \
  }                                                                                                \

CAST_INT2FLOAT_TEST_DEF(int, int2float_rd, FE_DOWNWARD)
CAST_INT2FLOAT_TEST_DEF(int, int2float_rn, FE_TONEAREST)
CAST_INT2FLOAT_TEST_DEF(int, int2float_ru, FE_UPWARD)
CAST_INT2FLOAT_TEST_DEF(int, int2float_rz, FE_TOWARDZERO)

CAST_INT2FLOAT_TEST_DEF(unsigned int, uint2float_rd, FE_DOWNWARD)
CAST_INT2FLOAT_TEST_DEF(unsigned int, uint2float_rn, FE_TONEAREST)
CAST_INT2FLOAT_TEST_DEF(unsigned int, uint2float_ru, FE_UPWARD)
CAST_INT2FLOAT_TEST_DEF(unsigned int, uint2float_rz, FE_TOWARDZERO)

CAST_INT2FLOAT_TEST_DEF(long long int, ll2float_rd, FE_DOWNWARD)
CAST_INT2FLOAT_TEST_DEF(long long int, ll2float_rn, FE_TONEAREST)
CAST_INT2FLOAT_TEST_DEF(long long int, ll2float_ru, FE_UPWARD)
CAST_INT2FLOAT_TEST_DEF(long long int, ll2float_rz, FE_TOWARDZERO)

CAST_INT2FLOAT_TEST_DEF(unsigned long long int, ull2float_rd, FE_DOWNWARD)
CAST_INT2FLOAT_TEST_DEF(unsigned long long int, ull2float_rn, FE_TONEAREST)
CAST_INT2FLOAT_TEST_DEF(unsigned long long int, ull2float_ru, FE_UPWARD)
CAST_INT2FLOAT_TEST_DEF(unsigned long long int, ull2float_rz, FE_TOWARDZERO)

CAST_KERNEL_DEF(int_as_float)

float int_as_float_ref(int arg) {
  float tmp;
  memcpy(&tmp, &arg, sizeof(tmp)); 
  return tmp;
}

TEST_CASE("Unit_Device_int_as_float_Positive") {
  float (*ref)(int) = int_as_float_ref;
  UnaryIntRangeTest(int_as_float_kernel<float, int>, ref, EqValidatorBuilderFactory<float>(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
}

CAST_KERNEL_DEF(uint_as_float)

float uint_as_float_ref(unsigned int arg) {
  float tmp;
  memcpy(&tmp, &arg, sizeof(tmp)); 
  return tmp;
}

TEST_CASE("Unit_Device_uint_as_float_Positive") {
   float (*ref)(unsigned int) = uint_as_float_ref;
  UnaryIntRangeTest(uint_as_float_kernel<unsigned int, float>, ref, EqValidatorBuilderFactory<float>(), std::numeric_limits<unsigned int>::lowest(), std::numeric_limits<unsigned int>::max());
}

CAST_KERNEL_DEF(double2loint)

int double2loint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp)); 
  return tmp[0];
}

TEST_CASE("Unit_Device_double2loint_Positive") {
  int (*ref)(double) = double2loint_ref;
  UnaryDoublePrecisionTest(double2loint_kernel<int, double>, ref, EqValidatorBuilderFactory<int>());
}


/*
  
 
#define CAST_DOUBLE2FLOAT_TEST_DEF(kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name)                                                                       \
                                                                                                   \
  float kern_name##_ref(double arg) {                                                              \
    int curr_direction = fegetround();                                                             \
    fesetround(round_dir);                                                                         \
    float result = static_cast<float>(arg);                                                        \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    UnaryDoublePrecisionTest(kern_name##_kernel<float, double>, ref,                               \
                             EqValidatorBuilderFactory<float>());                                  \
  }                                                                                                \

CAST_DOUBLE2LL_TEST_DEF(double2ll_rd, FE_DOWNWARD)
CAST_DOUBLE2LL_TEST_DEF(double2ll_rn, FE_TONEAREST)
CAST_DOUBLE2LL_TEST_DEF(double2ll_ru, FE_UPWARD)
CAST_DOUBLE2LL_TEST_DEF(double2ll_rz, FE_TOWARDZERO)

CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rd, FE_DOWNWARD)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rn, FE_TONEAREST)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_ru, FE_UPWARD)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rz, FE_TOWARDZERO)

CAST_DOUBLE2INT_TEST_DEF(unsigned long long int, double2ull_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(unsigned long long int, double2ull_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(unsigned long long int, double2ull_ru, FE_UPWARD)
CAST_DOUBLE2INT_TEST_DEF(unsigned long long int, double2ull_rz, FE_TOWARDZERO)
/*
CAST_DOUBLE2INT_TEST_DEF(long long int, double2ll_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(long long int, double2ll_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(long long int, double2ll_ru, FE_UPWARD)
CAST_DOUBLE2INT_TEST_DEF(long long int, double2ll_rz, FE_TOWARDZERO)*/
/*
CAST_KERNEL_DEF(int2float_rd)

TEST_CASE("Unit_int2float_rd_Positive") {
  auto int2float_ref = [](int arg) -> float {
    if (std::isnan(arg))
      return 0;
    else if (arg > std::numeric_limits<float>::max())
      return std::numeric_limits<float>::max();
    else if (arg < std::numeric_limits<float>::lowest())
      return std::numeric_limits<float>::lowest();
    return static_cast<float>(arg); };
  float (*ref)(int) = int2float_ref;
  UnaryIntRangeTest(double2int_rd_kernel<float, int>, ref, EqValidatorBuilderFactory<float>(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
}
/*

*/