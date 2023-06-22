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

#pragma once

#include "math_common.hh"
#include "math_special_values.hh"
#include "quaternary_common.hh"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>

namespace cg = cooperative_groups;

#define COMPLEX_UNARY_KERNEL_DEF(func_name)                                                        \
  template <typename T1, typename T2>                                                              \
  __global__ void func_name##_kernel(T1* const ys, const size_t num_xs, T2* const xs) {            \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<hipFloatComplex, T2>) {                                         \
        ys[i] = func_name##f(xs[i]);                                                               \
      } else if constexpr (std::is_same_v<hipDoubleComplex, T2>) {                                 \
        ys[i] = func_name(xs[i]);                                                                  \
      }                                                                                            \
    }                                                                                              \
  }

#define COMPLEX_BINARY_KERNEL_DEF(func_name)                                                       \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     T* const x2s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<hipFloatComplex, T>) {                                          \
        ys[i] = func_name##f(x1s[i], x2s[i]);                                                      \
      } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {                                  \
        ys[i] = func_name(x1s[i], x2s[i]);                                                         \
      }                                                                                            \
    }                                                                                              \
  }

#define COMPLEX_TERNARY_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<hipFloatComplex, T>) {                                          \
        ys[i] = func_name##f(x1s[i], x2s[i], x3s[i]);                                              \
      } else if constexpr (std::is_same_v<hipDoubleComplex, T>) {                                  \
        ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                 \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T1, typename T2, typename ValidatorBuilder>
void ComplexUnaryFloatingPointSpecialValuesTest(kernel_sig<T1, T2> kernel,
                                              ref_sig<T1, T2> ref_func,
                                              const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<decltype(T2().x)>>(kSpecialValReducedRegistry);

  const auto size = values.size * values.size;
  LinearAllocGuard<T2> xs{LinearAllocs::hipHostMalloc, size * sizeof(T2)};

  for (auto i = 0u; i < values.size; ++i) {
    for (auto j = 0u; j < values.size; ++j) {
      xs.ptr()[i * values.size + j].x = values.data[i];
      xs.ptr()[i * values.size + j].y = values.data[j];
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, xs.ptr());
}

template <typename T, typename ValidatorBuilder>
void ComplexBinaryFloatingPointSpecialValuesTest(kernel_sig<T, T, T> kernel,
                                              ref_sig<T, T, T> ref_func,
                                              const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<decltype(T().x)>>(kSpecialValReducedRegistry);

  const auto size = values.size * values.size * values.size * values.size;
  LinearAllocGuard<T> x1s{LinearAllocs::hipHostMalloc, size * sizeof(T)};
  LinearAllocGuard<T> x2s{LinearAllocs::hipHostMalloc, size * sizeof(T)};

  for (auto i = 0u; i < values.size; ++i) {
    for (auto j = 0u; j < values.size; ++j) {
      for (auto k = 0u; k < values.size; ++k) {
        for (auto l = 0u; l < values.size; ++l) {
          x1s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].x = values.data[i];
          x1s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].y = values.data[j];
          x2s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].x = values.data[k];
          x2s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].y = values.data[l];
        }
      }
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, x1s.ptr(),
                                x2s.ptr());
}

template <typename T, typename ValidatorBuilder>
void ComplexTernaryFloatingPointSpecialValuesTest(kernel_sig<T, T, T, T> kernel,
                                              ref_sig<T, T, T, T> ref_func,
                                              const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<decltype(T().x)>>(kSpecialValReducedRegistry);

  const auto size = values.size * values.size * values.size * values.size;
  LinearAllocGuard<T> x1s{LinearAllocs::hipHostMalloc, size * sizeof(T)};
  LinearAllocGuard<T> x2s{LinearAllocs::hipHostMalloc, size * sizeof(T)};
  LinearAllocGuard<T> x3s{LinearAllocs::hipHostMalloc, size * sizeof(T)};

  for (auto i = 0u; i < values.size; ++i) {
    for (auto j = 0u; j < values.size; ++j) {
      for (auto k = 0u; k < values.size; ++k) {
        for (auto l = 0u; l < values.size; ++l) {
          x1s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].x = values.data[i];
          x1s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].y = values.data[j];
          x2s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].x = values.data[k];
          x2s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].y = values.data[l];
          x3s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].x = values.data[l];
          x3s.ptr()[((i * values.size + j) * values.size + k) * values.size + l].y = values.data[k];
        }
      }
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, x1s.ptr(),
                                x2s.ptr(), x3s.ptr());
}

