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
  const uint64_t batch_size = n / thread_count;

  uint64_t begin, end = 0;
  for (uint64_t i = 0; i < thread_count; ++i) {
    begin = batch_size * i;
    end = begin + batch_size;
    thread_pool.Post([=] { task(begin, end); });
  }

  const uint64_t tail = n % thread_count;
  task(end, end + tail);

  thread_pool.Wait();
}

template <typename InputType>
__device__ void InputGeneratorImpl(ROCRAND_STATE* states, uint64_t* base, uint64_t* n, InputType* x,
                                   InputType (*wrapper)(ROCRAND_STATE*, uint64_t, uint64_t)) {
  // auto local_state = states[0];
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n[0]; i += blockDim.x * gridDim.x) {
    x[i] = wrapper(states, base[0], i);
  }
  // states[0] = local_state;
}

template <typename OutputType, typename InputType>
__device__ void TestValueGeneratorImpl(uint64_t* n, OutputType* y, InputType* x,
                                       OutputType (*wrapper)(InputType)) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n[0]; i += blockDim.x * gridDim.x) {
    y[i] = wrapper(x[i]);
  }
}

template <typename OutputType, typename InputType>
void ReferenceGeneratorImpl(uint64_t* n, OutputType* y, InputType* x,
                            OutputType (*wrapper)(InputType)) {
  const auto task = [=](uint64_t begin, uint64_t end) {
    for (auto i = begin; i < end; ++i) {
      y[i] = wrapper(x[i]);
    }
  };

  SubmitTasks(task, n[0]);
}

template <typename OutputType, typename InputType>
void ValidatorImpl(FailureReport* report, uint64_t* n, OutputType* y1, OutputType* y2, InputType* x,
                   bool (*wrapper)(OutputType, OutputType, InputType)) {
  const auto task = [=](uint64_t begin, uint64_t end) {
    static std::mutex mtx;
    for (auto i = begin; i < end; ++i) {
      // {
      //   std::lock_guard lg{mtx};
      //   std::cout << y1[i] << " " << y2[i] << " " << *reinterpret_cast<uint32_t*>(&x[i])
      //             << std::endl;
      // }
      if (!wrapper(y1[i], y2[i], x[i])) {
        // std::stringstream ss;
        // ss << std::scientific << std::setprecision(16) << &y1[i] << " " << &y2[i] << " " << &x[i]
        //    << " " << i << "\n";
        // // TODO improve report message

        // static std::mutex mtx;
        // {
        //   std::lock_guard lg{mtx};
        //   report->msg += ss.str();
        // }

        // return;
      }
    }
  };

  SubmitTasks(task, n[0]);
}

template <typename OutputType, typename InputType>
void MathTestImpl(void (*input_generator)(ROCRAND_STATE*, uint64_t*, uint64_t*, InputType*),
                  void (*test_value_generator)(uint64_t*, OutputType*, InputType*),
                  void (*reference_generator)(void**), void (*validator)(void**),
                  uint64_t batch_size, uint64_t begin, uint64_t end) {
  GraphEngineParams params = {0};
  params.input_generator = reinterpret_cast<void*>(input_generator);
  params.test_value_generator = reinterpret_cast<void*>(test_value_generator);
  params.reference_generator = reinterpret_cast<void*>(reference_generator);
  params.validator = reinterpret_cast<void*>(validator);
  params.batch_size = batch_size;

  GraphEngine<OutputType, InputType> graph_engine{params};
  {
    StreamGuard stream{Streams::perThread};
    graph_engine.Launch(begin, end, stream.stream());
  }
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