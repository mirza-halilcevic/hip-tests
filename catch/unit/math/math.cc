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

__global__ void sin_kernel(double* const results, const size_t num_xs, double* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid < num_xs) {
    results[tid] = sin(xs[tid]);
  }
}

TEST_CASE("Sin") {
  double xs[] = {0., 1., 2., 3.14159};
  MathTest(ULPValidator{2}, sin_kernel, sin, 4, xs).Run(1u, 4u);
}

__global__ void atan2_kernel(double* const results, const size_t num_xs, double* const x1s,
                             double* const x2s) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid < num_xs) {
    results[tid] = atan2(x1s[tid], x2s[tid]);
  }
}

TEST_CASE("Atan2") {
  double x1s[] = {0., 1., 2., 3.14159};
  double x2s[] = {0., 1., 2., 3.14159};
  MathTest(ULPValidator{2}, atan2_kernel, atan2, 4, x1s, x2s).Run(1u, 4u);
}