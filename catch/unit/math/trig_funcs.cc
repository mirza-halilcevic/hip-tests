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

template <typename F, typename RF, typename ValidatorBuilder>
void UnarySinglePrecisionBruteForceTest(F kernel, RF ref_func,
                                        const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = grid_size * block_size;
  std::vector<float> values(max_batch_size);

  MathTest<float, double, 1> math_test(max_batch_size);

  uint64_t stop = std::numeric_limits<uint32_t>::max() + 1ul;
  auto batch_size = max_batch_size;
  uint32_t val = 0u;
  const auto num_threads = thread_pool.thread_count();
  for (uint64_t v = 0u; v < stop;) {
    batch_size = std::min<uint64_t>(max_batch_size, stop - v);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);

      thread_pool.Post([=, &values] {
        auto t = v;
        uint32_t val;
        for (auto j = 0u; j < sub_batch_size; ++j) {
          val = static_cast<uint32_t>(t++);
          values[base_idx + j] = *reinterpret_cast<float*>(&val);
        }
      });

      v += sub_batch_size;
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    // for (auto i = 0u; i < batch_size; ++i) {
    //   val = static_cast<uint32_t>(v);
    //   values[i] = *reinterpret_cast<float*>(&val);
    //   ++v;
    // }

    math_test.Run(validator_builder, grid_size, block_size, kernel, ref_func, batch_size,
                  values.data());
  }
}

template <typename F, typename RF, typename ValidatorBuilder>
void UnaryDoublePrecisionTest(F kernel, RF ref_func, const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = grid_size * block_size;
  std::vector<double> values(max_batch_size);

  MathTest<double, long double, 1> math_test(max_batch_size);

  const uint64_t num_args = std::numeric_limits<uint32_t>::max();
  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();
  for (uint64_t i = 0ul; i < num_args; i += batch_size) {
    batch_size = std::min<uint64_t>(max_batch_size, num_args - i);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);
      thread_pool.Post([=, &values] {
        const auto generator = [] {
          static thread_local std::default_random_engine rng(std::random_device{}());
          std::uniform_real_distribution<double> unif_dist(std::numeric_limits<double>::lowest(),
                                                           std::numeric_limits<double>::max());
          return unif_dist(rng);
        };
        std::generate(values.begin() + base_idx, values.begin() + base_idx + sub_batch_size,
                      generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, kernel, ref_func, batch_size,
                  values.data());
  }
}

template <typename T, typename F, typename RF, typename ValidatorBuilder>
void UnarySpecialValuesTest(F kernel, RF ref_func, const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = grid_size * block_size;

  const auto values = std::get<SpecialVals<T>>(kSpecialValRegistry);

  MathTest<T, RefType_t<T>, 1> math_test(max_batch_size);

  auto batch_size = max_batch_size;
  for (uint64_t i = 0u; i < values.size; i += batch_size) {
    batch_size = std::min<uint64_t>(max_batch_size, values.size - i);
    math_test.Run(validator_builder, grid_size, block_size, kernel, ref_func, batch_size,
                  values.data);
  }
}

TEMPLATE_TEST_CASE("Sin", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = std::sin;
  const auto validator_builder = ULPValidatorBuilderFactory<TestType>(2);

  SECTION("Brute force") {
    if constexpr (std::is_same_v<float, TestType>) {
      UnarySinglePrecisionBruteForceTest(sin_kernel<TestType>, ref, validator_builder);
    } else {
      UnaryDoublePrecisionTest(sin_kernel<double>, ref, validator_builder);
    }
  }

  SECTION("Special values") {
    UnarySpecialValuesTest<TestType>(sin_kernel<TestType>, ref, validator_builder);
  }
}

// MATH_DOUBLE_ARG_KERNEL_DEF(atan2)

// TEST_CASE("Atan2") {
//   float x1s[] = {0.f, 1.f, 2.f, 3.14159f};
//   float x2s[] = {0.f, 1.f, 2.f, 3.14159f};
//   double (*ref)(double, double) = atan2;
//   MathTest(ULPValidator{2}, 1u, 4u, atan2_kernel<float>, ref, 4u, x1s, x2s);
// }