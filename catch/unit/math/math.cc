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

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename Validator, typename T, typename... Ts> class MathTest {
  static_assert(std::conjunction_v<std::is_same<T, Ts>...>, "Message");
  using kernel_sig = void (*)(T*, const size_t, Ts*...);
  using reference_sig = T (*)(Ts...);

  class LAWrapper {
   public:
    LAWrapper(const size_t size, T* const init_vals) : la_{LinearAllocs::hipMalloc, size, 0u} {
      HIP_CHECK(hipMemcpy(la_.ptr(), init_vals, size, hipMemcpyHostToDevice));
    }

    T* ptr() { return la_.ptr(); }

   private:
    LinearAllocGuard<T> la_;
  };

 public:
  MathTest(Validator validator, kernel_sig kernel, reference_sig ref_func, size_t num_args,
           Ts*... xss)
      : validator_{validator},
        kernel_{kernel},
        ref_func_{ref_func},
        num_args_{num_args},
        xss_{xss...},
        xss_dev_{LAWrapper(num_args * sizeof(T), xss)...} {}

  void Run(const size_t grid_dim, const size_t block_dim) {
    RunImpl(grid_dim, block_dim, std::index_sequence_for<Ts...>{});
  }

 private:
  template <size_t... I>
  void RunImpl(const size_t grid_dim, const size_t block_dim, std::index_sequence<I...>) {
    // An input dev array could be used to store the results to reduce memory usage
    LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, num_args_ * sizeof(T)};
    kernel_<<<grid_dim, block_dim>>>(y_dev.ptr(), num_args_, xss_dev_[I].ptr()...);
    HIP_CHECK(hipGetLastError());

    // An input array could be reused to store the results to reduce memory usage
    std::vector<T> y(num_args_);
    HIP_CHECK(hipMemcpy(y.data(), y_dev.ptr(), num_args_ * sizeof(T), hipMemcpyDeviceToHost));

    // The host could calculate reference values after the kernel is launched, even in parallel on
    // several threads, to accelerate test execution. This would require allocating another array,
    // but if the above specified memory reuse is performed, the footprint would remain unchanged.
    for (auto i = 0u; i < num_args_; ++i) {
      validator_.validate(y[i], ref_func_(xss_[I][i]...));
    }
  }

  Validator validator_;
  kernel_sig kernel_;
  reference_sig ref_func_;
  size_t num_args_;
  std::array<T*, sizeof...(Ts)> xss_;
  std::array<LAWrapper, sizeof...(Ts)> xss_dev_;
};


template <typename T, typename RT, typename V>
void Foo(void (*kernel)(T*, size_t, T*), RT (*ref_func)(RT), size_t num_args, T* xs, V validator,
         size_t grid_dim, size_t block_dim) {
  LinearAllocGuard<T> xs_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(xs_dev.ptr(), xs, num_args * sizeof(T), hipMemcpyHostToDevice));

  LinearAllocGuard<T> ys_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(ys_dev.ptr(), num_args, xs_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> ys(num_args);
  HIP_CHECK(hipMemcpy(ys.data(), ys_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    validator.validate(ys[i], static_cast<T>(ref_func(static_cast<RT>(xs[i]))));
  }
}

template <typename T, typename RT, typename V>
void Foo(void (*kernel)(T*, size_t, T*, T*), RT (*ref_func)(RT, RT), size_t num_args, T* x1s,
         T* x2s, V validator, size_t grid_dim, size_t block_dim) {
  LinearAllocGuard<T> x1s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x1s_dev.ptr(), x1s, num_args * sizeof(T), hipMemcpyHostToDevice));
  LinearAllocGuard<T> x2s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x2s_dev.ptr(), x2s, num_args * sizeof(T), hipMemcpyHostToDevice));

  LinearAllocGuard<T> ys_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(ys_dev.ptr(), num_args, x1s_dev.ptr(), x2s_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> ys(num_args);
  HIP_CHECK(hipMemcpy(ys.data(), ys_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    validator.validate(ys[i],
                       static_cast<T>(ref_func(static_cast<RT>(x1s[i]), static_cast<RT>(x2s[i]))));
  }
}

struct ULPValidator {
  template <typename T> void validate(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinULP(ref_val, ulps));
  }

  const int64_t ulps;
};

struct AbsValidator {
  template <typename T> void validate(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinAbs(ref_val, margin));
  }

  const double margin;
};

template <typename T> struct RelValidator {
  void validate(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinRel(ref_val, margin));
  }

  const T margin;
};

// Can be used for integer functions as well
struct EqValidator {
  template <typename T> void validate(const T actual_val, const T ref_val) const {
    REQUIRE(actual_val == ref_val);
  }
};

__global__ void sin_kernel(float* const results, const size_t num_xs, float* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid < num_xs) {
    results[tid] = sinf(xs[tid]);
  }
}

TEST_CASE("Sin") {
  float xs[] = {0.f, 1.f, 2.f, 3.14159f};
  Foo<float, double>(sin_kernel, sin, 4, xs, ULPValidator{2}, 1u, 4u);
  // MathTest(ULPValidator{2}, sin_kernel, sin, 4, xs).Run(1u, 4u);
}

__global__ void atan2_kernel(float* const results, const size_t num_xs, float* const x1s,
                             float* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid < num_xs) {
    results[tid] = atan2f(x1s[tid], x2s[tid]);
  }
}

TEST_CASE("Atan2") {
  float x1s[] = {0.f, 1.f, 2.f, 3.14159f};
  float x2s[] = {0.f, 1.f, 2.f, 3.14159f};
  Foo<float, double>(atan2_kernel, atan2, 4, x1s, x2s, ULPValidator{2}, 1u, 4u);
  // MathTest(ULPValidator{2}, atan2_kernel, atan2, 4, x1s, x2s).Run(1u, 4u);
}