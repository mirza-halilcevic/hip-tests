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
    const auto blocks = GenerateBlockDimensionsForShuffle();
    // const auto blocks = dim3(2, 1, 1);
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensionsForShuffle();
    // const auto threads = dim3(40);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);
    width_ = generate_width();
    // width_ = 4;
    INFO("Width: " << width_);

    const auto alloc_size = grid_.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    // Allocate active mask array on host and device
    // Generate active masks

    cast_to_derived().launch_kernel(arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    cast_to_derived().validate(arr.ptr());
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
  CPUGrid grid_;
  int width_;
};

namespace cg = cooperative_groups;

template <typename T>
__global__ void shfl_up(T* const out, const unsigned int delta, const int width) {
  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % 32);
  out[grid.thread_rank()] = __shfl_up(var, delta, width);
}

template <typename T> class ShflUp : public Foo<ShflUp<T>, T> {
 public:
  void launch_kernel(T* const arr_dev) {
    delta_ = GENERATE_COPY(range(0, this->width_));
    INFO("Delta: " << delta_);
    shfl_up<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, delta_, this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> T {
      const int rank_in_partition = this->grid_.thread_rank_in_block(i).value() % this->width_;
      const int target = rank_in_partition - delta_;
      return (target < 0 ? i : i - delta_) % 32;
    });
  };

 private:
  unsigned int delta_;
};

TEMPLATE_TEST_CASE("ShflUp", "", int, float) { ShflUp<TestType>().run(); }


template <typename T>
__global__ void shfl_down(T* const out, const unsigned int delta, const int width) {
  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % 32);
  out[grid.thread_rank()] = __shfl_down(var, delta, width);
}

template <typename T> class ShflDown : public Foo<ShflDown<T>, T> {
 public:
  void launch_kernel(T* const arr_dev) {
    delta_ = GENERATE_COPY(range(0, this->width_));
    INFO("Delta: " << delta_);
    shfl_down<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, delta_, this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const int rank_in_block = this->grid_.thread_rank_in_block(i).value();
      if (rank_in_block + this->width_ >= this->grid_.threads_in_block_count_) {
        return std::nullopt;
      }

      const int rank_in_partition = rank_in_block % this->width_;
      const int target = rank_in_partition + delta_;
      return (target >= this->width_ ? i : i + delta_) % 32;
    });
  };

 private:
  unsigned int delta_;
};

TEMPLATE_TEST_CASE("ShflDown", "", int, float) { ShflDown<TestType>().run(); }


template <typename T>
__global__ void shfl(T* const out, const uint8_t* const src_lanes, const int width) {
  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % 32);
  out[grid.thread_rank()] = __shfl(var, src_lanes[cg::this_thread_block() % width], width);
}

template <typename T> class Shfl : public Foo<Shfl<T>, T> {
 public:
  void launch_kernel(T* const arr_dev) {
    // TODO Reconsider how mask is generated
    lane_mask_ = GENERATE_COPY(range(0, this->warp_size_));
    // lane_mask_ = 1;
    // std::cout << "Lane mask: " << lane_mask_ << std::endl;
    INFO("Lane mask: " << lane_mask_);
    shfl_xor<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, lane_mask_, this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_warp = this->grid_.thread_rank_in_block(i).value() % this->warp_size_;
      const int warp_target = rank_in_warp ^ this->lane_mask_;
      const auto target = i + warp_target - rank_in_warp;

      if (target >= this->grid_.threads_in_block_count_) {
        return std::nullopt;
      }

      const auto target_partition = warp_target / this->width_;
      const auto partition_rank = rank_in_warp / this->width_;
      return (target_partition > partition_rank ? i : target) % 32;
    });
  };

 private:
  int lane_mask_;
};

TEMPLATE_TEST_CASE("Shfl", "", int, float) { Shfl<TestType>().run(); }