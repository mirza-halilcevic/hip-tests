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

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

#include <bitset>
#include <array>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

template <typename Derived, typename T> class Foo {
 public:
  Foo() : warp_size_{get_warp_size()} {}

  void run() {
    // Generate blocks
    blocks_ = GenerateBlockDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks_.x << ", y " << blocks_.y << ", z " << blocks_.z);
    // Generate threads
    threads_ = GenerateThreadDimensionsForShuffle();
    INFO("Block dimensions: x " << threads_.x << ", y " << threads_.y << ", z " << threads_.z);
    // Generate width
    width_ = generate_width();
    INFO("Width: " << width_);

    // Create CPU grid
    // Allocate output arr of type T on host and device
    // Allocate active mask array on host and device
    // Generate active masks
    cast_to_derived().launch_kernel(/*Pass in dev mask array and dev output array*/);
    // Copy output arr from dev to host
    cast_to_derived().validate(/*Pass in host output arr*/);
  }

 private:
  int get_warp_size() const {
    int current_dev = -1;
    HIP_CHECK(hipGetDevice(&current_dev));
    int warp_size = 0u;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    return warp_size;
  }

  int generate_width() const {
    int exponent = 0;
    int warp_size = warp_size_;
    while (warp_size >>= 1) {
      ++exponent;
    }

    return GENERATE_COPY(map([](int e) { return 1 << e; }, range(1, exponent + 1)));
  }

  Derived& cast_to_derived() { return reinterpret_cast<Derived&>(*this); }

 protected:
  const int warp_size_;
  dim3 blocks_;
  dim3 threads_;
  int width_;
};

template <typename T> class Bar : public Foo<Bar<T>, T> {
 public:
  void launch_kernel() { std::cout << this->width_ << std::endl; }

  void validate() {};

 private:
};

TEST_CASE("Blahem") {
  Bar<int> b;
  b.run();
}