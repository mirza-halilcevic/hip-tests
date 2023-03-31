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

  SECTION("Special values") {
    const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
    MathTest<false>(ULPValidatorGenerator<TestType>(2), 1u, special_vals.size, sin_kernel<TestType>,
                    ref, special_vals.size, special_vals.data);
  }

  SECTION("Brute force") {
    // int sm_count = 0;
    // HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, 0));

    int grid_size = 0, block_size = 0;
    HIP_CHECK(
        hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size, sin_kernel<TestType>, 0, 0));
    std::cout << grid_size << " " << block_size << std::endl;
    std::vector<TestType> values(grid_size * block_size);
    size_t inserted = 0u;
    constexpr TestType pi = M_PI;
    constexpr TestType delta = 1.;

    const auto validator = ULPValidatorGenerator<TestType>(2);
    for (TestType v = 0; v <= 1'000'000'000.f; v += delta) {
      values[inserted++] = v;
      if (inserted < values.size()) continue;
      MathTest<true>(validator, grid_size, block_size, sin_kernel<TestType>, ref, inserted,
                     values.data());
      inserted = 0u;
    }

    if (inserted != 0) {
      MathTest<true>(validator, grid_size, block_size, sin_kernel<TestType>, ref, inserted,
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