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

#include "math_common.hh"
#include "math_special_values.hh"

MATH_SINGLE_ARG_KERNEL_DEF(sin)

TEMPLATE_TEST_CASE("Sin", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = std::sin;

  SECTION("Brute force") {
    const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(sin_kernel<TestType>);
    const uint32_t max_batch_size = grid_size * block_size;
    std::vector<TestType> values(max_batch_size);

    MathTest<TestType, T, 1> math_test(max_batch_size);

    const auto validator_builder = ULPValidatorBuilderFactory<TestType>(2);

    uint32_t stop = std::numeric_limits<uint32_t>::max();
    uint32_t batch_size = max_batch_size;
    for (uint32_t v = 0u; v != stop;) {
      batch_size = std::min(max_batch_size, stop - v);

      for (uint32_t i = 0u; i < batch_size; ++i) {
        values[i] = *reinterpret_cast<float*>(&v);
        ++v;
      }

      math_test.Run(validator_builder, grid_size, block_size, sin_kernel<TestType>, ref, batch_size,
                    values.data());
    }
  }
}

// MATH_DOUBLE_ARG_KERNEL_DEF(atan2)

// TEST_CASE("Atan2") {
//   float x1s[] = {0.f, 1.f, 2.f, 3.14159f};
//   float x2s[] = {0.f, 1.f, 2.f, 3.14159f};
//   double (*ref)(double, double) = atan2;
//   MathTest(ULPValidator{2}, 1u, 4u, atan2_kernel<float>, ref, 4u, x1s, x2s);
// }