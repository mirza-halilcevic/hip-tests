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

#include "helpers.hh"

#define INPUT_GENERATOR(func_name) InputGenerator_##func_name
#define INPUT_GENERATOR_WRAPPER(func_name) InputGenerator_##func_name##_Wrapper

#define INPUT_GENERATOR_WRAPPER_DEF(func_name)                                                     \
  __device__ INPUT_TYPE INPUT_GENERATOR_WRAPPER(func_name)(rocrand_state_xorwow * states,          \
                                                           uint64_t i)

#define INPUT_GENERATOR_DEF(func_name)                                                             \
  template <bool brute_force>                                                                      \
  __global__ void INPUT_GENERATOR(func_name)(rocrand_state_xorwow * states, uint64_t * n,          \
                                             INPUT_TYPE * x) {                                     \
    InputGeneratorImpl<brute_force>(states, n, x, INPUT_GENERATOR_WRAPPER(func_name));             \
  }

#define TEST_VALUE_GENERATOR(func_name) TestValueGenerator_##func_name
#define TEST_VALUE_GENERATOR_WRAPPER(func_name) TestValueGenerator_##func_name##_Wrapper

#define TEST_VALUE_GENERATOR_WRAPPER_DEF(func_name)                                                \
  __device__ OUTPUT_TYPE TEST_VALUE_GENERATOR_WRAPPER(func_name)(INPUT_TYPE x)

#define TEST_VALUE_GENERATOR_DEF(func_name)                                                        \
  template <bool brute_force>                                                                      \
  __global__ void TEST_VALUE_GENERATOR(func_name)(uint64_t * n, OUTPUT_TYPE * y, INPUT_TYPE * x) { \
    TestValueGeneratorImpl<brute_force>(n, y, x, TEST_VALUE_GENERATOR_WRAPPER(func_name));         \
  }

#define REFERENCE_GENERATOR(func_name) ReferenceGenerator_##func_name
#define REFERENCE_GENERATOR_WRAPPER(func_name) ReferenceGenerator_##func_name##_Wrapper

#define REFERENCE_GENERATOR_WRAPPER_DEF(func_name)                                                 \
  OUTPUT_TYPE REFERENCE_GENERATOR_WRAPPER(func_name)(INPUT_TYPE x)

#define REFERENCE_GENERATOR_DEF(func_name)                                                         \
  template <bool brute_force> void REFERENCE_GENERATOR(func_name)(void** args) {                   \
    const auto arg0 = reinterpret_cast<uint64_t**>(args[0]);                                       \
    const auto arg1 = reinterpret_cast<OUTPUT_TYPE**>(args[1]);                                    \
    const auto arg2 = reinterpret_cast<INPUT_TYPE**>(args[2]);                                     \
    ReferenceGeneratorImpl<brute_force>(*arg0, *arg1, *arg2,                                       \
                                        REFERENCE_GENERATOR_WRAPPER(func_name));                   \
  }

#define VALIDATOR(func_name) Validator_##func_name
#define VALIDATOR_WRAPPER(func_name) Validator_##func_name##_Wrapper

#define VALIDATOR_WRAPPER_DEF(func_name)                                                           \
  bool VALIDATOR_WRAPPER(func_name)(OUTPUT_TYPE y1, OUTPUT_TYPE y2, INPUT_TYPE x)

#define VALIDATOR_DEF(func_name)                                                                   \
  template <bool brute_force> void VALIDATOR(func_name)(void** args) {                             \
    const auto arg0 = reinterpret_cast<std::string*>(args[0]);                                     \
    const auto arg1 = reinterpret_cast<uint64_t**>(args[1]);                                       \
    const auto arg2 = reinterpret_cast<OUTPUT_TYPE**>(args[2]);                                    \
    const auto arg3 = reinterpret_cast<OUTPUT_TYPE**>(args[3]);                                    \
    const auto arg4 = reinterpret_cast<INPUT_TYPE**>(args[4]);                                     \
    ValidatorImpl<brute_force>(*arg0, *arg1, *arg2, *arg3, *arg4, VALIDATOR_WRAPPER(func_name));   \
  }

#define MATH_TEST_BRUTE_FORCE(func_name, base_val, batch_size, num_val)                            \
  MathTestImpl<true>(+INPUT_GENERATOR(func_name) < true >,                                         \
                     +TEST_VALUE_GENERATOR(func_name) < true >,                                    \
                     +REFERENCE_GENERATOR(func_name) < true >, +VALIDATOR(func_name) < true >,     \
                     base_val, batch_size, num_val)

#define MATH_TEST_RANDOM(func_name, batch_size, num_val)                                           \
  MathTestImpl<false>(+INPUT_GENERATOR(func_name) < false >,                                       \
                      +TEST_VALUE_GENERATOR(func_name) < false >,                                  \
                      +REFERENCE_GENERATOR(func_name) < false >, +VALIDATOR(func_name) < false >,  \
                      0UL, batch_size, num_val)