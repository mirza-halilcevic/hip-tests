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

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>

#include <hip_test_common.hh>

#include "thread_pool.hh"

template <typename T>
using IntegerType = std::conditional_t<std::is_same_v<T, float>, uint32_t, uint64_t>;

template <typename FloatingPoint, typename Integer>
__host__ __device__ FloatingPoint raw_add(FloatingPoint base, Integer i) {
  Integer base_i = *reinterpret_cast<Integer*>(&base) + i;
  return *reinterpret_cast<FloatingPoint*>(&base_i);
}

template <typename T> void SubmitTasks(T&& task, uint64_t n) {
  const uint64_t thread_count = thread_pool.thread_count();
  const uint64_t chunk_size = n / thread_count;
  const uint64_t tail_size = n % thread_count;

  uint64_t begin, end;
  for (uint64_t i = 0; i < thread_count; ++i) {
    begin = chunk_size * i;
    end = begin + chunk_size;
    thread_pool.Post([=] { task(begin, end); });
  }

  task(end, end + tail_size);

  thread_pool.Wait();
}

template <bool brute_force, typename T>
__device__ void InputGeneratorImpl(rocrand_state_xorwow* states, uint64_t* n, T* x,
                                   T (*wrapper)(rocrand_state_xorwow*, uint64_t)) {
  if constexpr (brute_force) {
    static_assert(std::is_floating_point_v<T>);
    x[0] = *reinterpret_cast<T*>(&n[0]);
  } else {
    rocrand_state_xorwow local_state = states[0];
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n[0];
         i += blockDim.x * gridDim.x) {
      x[i] = wrapper(&local_state, i);
    }
    states[0] = local_state;
  }
}

template <bool brute_force, typename T, typename U>
__device__ void TestValueGeneratorImpl(uint64_t* n, T* y, U* x, T (*wrapper)(U)) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n[0]; i += blockDim.x * gridDim.x) {
    if constexpr (brute_force) {
      static_assert(std::is_floating_point_v<T>);
      static_assert(std::is_floating_point_v<U>);
      y[i] = wrapper(raw_add(x[0], static_cast<IntegerType<U>>(i)));
    } else {
      y[i] = wrapper(x[i]);
    }
  }

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (tid == 0) printf("%d\n", n[0]);
}

template <bool brute_force, typename T, typename U>
void ReferenceGeneratorImpl(uint64_t* n, T* y, U* x, T (*wrapper)(U)) {
  const auto task = [=](uint64_t begin, uint64_t end) {
    for (uint64_t i = begin; i < end; ++i) {
      if constexpr (brute_force) {
        static_assert(std::is_floating_point_v<T>);
        static_assert(std::is_floating_point_v<U>);
        y[i] = wrapper(raw_add(x[0], static_cast<IntegerType<U>>(i)));
      } else {
        y[i] = wrapper(x[i]);
      }
    }
  };

  SubmitTasks(task, n[0]);
}

template <bool brute_force, typename T, typename U>
void ValidatorImpl(std::string& report, uint64_t* n, T* y1, T* y2, U* x, bool (*wrapper)(T, T, U)) {
  const auto task = [=, &report](uint64_t begin, uint64_t end) {
    for (uint64_t i = begin; i < end; ++i) {
      U input;
      if constexpr (brute_force) {
        static_assert(std::is_floating_point_v<T>);
        static_assert(std::is_floating_point_v<U>);
        input = x[0];
      } else {
        input = x[i];
      }

      if (!wrapper(y1[i], y2[i], input)) {
        std::stringstream ss;
        ss << std::scientific << std::setprecision(16) << y1[i] << " " << y2[i] << " " << input
           << "\n";
        // TODO improve report message

        static std::mutex mtx;
        {
          std::lock_guard lg{mtx};
          report += ss.str();
        }

        return;
      }
    }
  };

  SubmitTasks(task, n[0]);
}

template <bool brute_force, typename T, typename U>
void MathTestImpl(void (*input_generator)(rocrand_state_xorwow*, uint64_t*, U*),
                  void (*test_value_generator)(uint64_t*, T*, U*),
                  void (*reference_generator)(void**), void (*validator)(void**),
                  uint64_t base_value, uint64_t batch_size, uint64_t num_values) {
  GraphEngineParams params = {0};
  params.input_generator = reinterpret_cast<void*>(input_generator);
  params.test_value_generator = reinterpret_cast<void*>(test_value_generator);
  params.reference_generator = reinterpret_cast<void*>(reference_generator);
  params.validator = reinterpret_cast<void*>(validator);
  params.input_sizeof = sizeof(U);
  params.test_value_sizeof = sizeof(T);
  params.base_value = base_value;
  params.batch_size = batch_size;

  GraphEngine<brute_force> graph_engine{params};

  StreamGuard stream{Streams::created};
  graph_engine.Launch(num_values, stream.stream());
}

template <typename T, typename Matcher> class ValidatorBase : public Catch::MatcherBase<T> {
 public:
  template <typename... Ts>
  ValidatorBase(T target, Ts&&... args) : matcher_{std::forward<Ts>(args)...}, target_{target} {}

  bool match(const T& val) const override {
    if (std::isnan(target_)) {
      return std::isnan(val);
    }

    return matcher_.match(val);
  }

  virtual std::string describe() const override {
    if (std::isnan(target_)) {
      return "is not NaN";
    }

    return matcher_.describe();
  }

 private:
  Matcher matcher_;
  T target_;
  bool nan = false;
};

template <typename T> auto ULPValidatorBuilderFactory(int64_t ulps) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinUlpsMatcher>{
        target, Catch::WithinULP(target, ulps)};
  };
};

template <typename T> auto AbsValidatorBuilderFactory(double margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinAbsMatcher>{
        target, Catch::WithinAbs(target, margin)};
  };
}

template <typename T> auto RelValidatorBuilderFactory(T margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinRelMatcher>{
        target, Catch::WithinRel(target, margin)};
  };
}