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

template <typename T> class LAWrapper {
 public:
  LAWrapper(const size_t size, T* const init_vals) : la_{LinearAllocs::hipMalloc, size, 0u} {
    HIP_CHECK(hipMemcpy(la_.ptr(), init_vals, size, hipMemcpyHostToDevice));
  }

  T* ptr() { return la_.ptr(); }

 private:
  LinearAllocGuard<T> la_;
};

// void Foo(void (*kernel)(T*, const size_t, Ts*...), T (*reference_func)(Ts...), size_t num_args,
//          Ts*... args) {
//   std::array<LAWrapper<T>, sizeof...(Ts)> xs_dev{LAWrapper(num_args * sizeof(T), args)...};
//   // head(xs) could be reused for output, reducing memory usage
//   LinearAllocGuard<T> y_dev(LinearAllocs::hipMalloc, num_args * sizeof(T));
//   // LinearAllocGuards xs_dev(LinearAllocs::hipMalloc, num_args * sizeof(T));
//   // LinearAllocGuards results_dev(LinearAllocs::hipMalloc, num_args*sizeof(T));
//   // HIP_CHECK(hipMemcpy(xs_dev.ptr(), ))
// }


template <typename T, typename... Ts> class MathTest {
  static_assert(std::conjunction_v<std::is_same<T, Ts>...>, "Message");
  using kernel_sig = void (*)(T*, const size_t, Ts*...);
  using reference_sig = T (*)(Ts...);

 public:
  MathTest(kernel_sig kernel, reference_sig ref_func, size_t num_args, Ts*... args)
      : kernel_{kernel},
        ref_func_{ref_func},
        num_args_{num_args},
        xs_dev_{LAWrapper(num_args * sizeof(T), args)...},
        y_dev_{LinearAllocs::hipMalloc, num_args * sizeof(T)} {}

  void Run() { 
    LaunchKernel(std::index_sequence_for<Ts...>{}); 
    std::vector<T> y(num_args_);
    HIP_CHECK(hipMemcpy(y.data(), y_dev_.ptr(), num_args_ * sizeof(T), hipMemcpyDeviceToHost));
  }


 private:
  template <size_t... I> void LaunchKernel(std::index_sequence<I...>) {
    kernel_<<<1, num_args_>>>(y_dev_.ptr(), num_args_, xs_dev_[I].ptr()...);
    HIP_CHECK(hipGetLastError());
  }

  kernel_sig kernel_;
  reference_sig ref_func_;
  size_t num_args_;
  std::array<LAWrapper<T>, sizeof...(Ts)> xs_dev_;
  LinearAllocGuard<T> y_dev_;
};

__global__ void sin_kernel(double* const results, const size_t num_xs, double* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  printf("Hello\n");
  results[tid] = sin(xs[tid]);
}

TEST_CASE("Foo") {
  double arg = 0;
  MathTest(sin_kernel, sin, 1, &arg).Run();
}