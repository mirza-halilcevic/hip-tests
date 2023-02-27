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

#include <bitset>
#include <array>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

static int generate_width(int warp_size) {
  int exponent = 0;
  while (warp_size >>= 1) {
    ++exponent;
  }

  return GENERATE_COPY(map([](int e) { return 1 << e; }, range(1, exponent + 1)));
}

template <typename T>
__global__ void shfl_up(T* const out, const uint64_t* const active_masks, const unsigned int delta,
                        const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % warpSize);
  out[grid.thread_rank()] = __shfl_up(var, delta, width);
}

template <typename T> class ShflUp : public WarpTest<ShflUp<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    delta_ = GENERATE_COPY(range(0, width_));
    INFO("Delta: " << delta_);
    shfl_up<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, delta_,
                                                               width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % width_ - delta_;
      if (!active_mask.test(rank_in_warp) ||
          (target >= 0 && !active_mask.test(rank_in_warp - delta_))) {
        return std::nullopt;
      }

      return (target < 0 ? i : i - delta_) % this->warp_size_;
    });
  };

 private:
  unsigned int delta_;
  int width_;
};

TEMPLATE_TEST_CASE("Unit_Warp_Functions_Shfl_Up_Positive_Basic", "", int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  ShflUp<TestType>().run();
}


template <typename T>
__global__ void shfl_down(T* const out, const uint64_t* const active_masks,
                          const unsigned int delta, const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % warpSize);
  out[grid.thread_rank()] = __shfl_down(var, delta, width);
}

template <typename T> class ShflDown : public WarpTest<ShflDown<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    delta_ = GENERATE_COPY(range(0, width_));
    INFO("Delta: " << delta_);
    shfl_down<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, delta_,
                                                                 width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const int rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % width_ + delta_;

      if (!active_mask.test(rank_in_warp) ||
          (target < width_ && !active_mask.test(rank_in_warp + delta_)) ||
          (rank_in_block + delta_ >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (target >= width_ ? i : i + delta_) % this->warp_size_;
    });
  };

 private:
  unsigned int delta_;
  int width_;
};

TEMPLATE_TEST_CASE("Unit_Warp_Functions_Shfl_Down_Positive_Basic", "", int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  ShflDown<TestType>().run();
}


template <typename T>
__global__ void shfl_xor(T* const out, const uint64_t* const active_masks, const int lane_mask,
                         const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % warpSize);
  out[grid.thread_rank()] = __shfl_xor(var, lane_mask, width);
}

template <typename T> class ShflXOR : public WarpTest<ShflXOR<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    lane_mask_ = GENERATE_COPY(range(0, this->warp_size_));
    INFO("Lane mask: " << lane_mask_);
    shfl_xor<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, lane_mask_,
                                                                width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const int warp_target = rank_in_warp ^ this->lane_mask_;
      const int target_offset = warp_target - rank_in_warp;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const auto target_partition = warp_target / width_;
      const auto partition_rank = rank_in_warp / width_;
      if (!active_mask.test(rank_in_warp) ||
          (target_partition <= partition_rank && !active_mask.test(rank_in_warp + target_offset)) ||
          (rank_in_block + target_offset >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (target_partition > partition_rank ? i : i + target_offset) % this->warp_size_;
    });
  };

 private:
  int lane_mask_;
  int width_;
};

TEMPLATE_TEST_CASE("Unit_Warp_Functions_Shfl_Xor_Positive_Basic", "", int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  ShflXOR<TestType>().run();
}


template <typename T>
__global__ void shfl(T* const out, const uint64_t* const active_masks,
                     const uint8_t* const src_lanes, const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }
  const auto grid = cg::this_grid();
  const auto block = cg::this_thread_block();
  T var = static_cast<T>(grid.thread_rank() % warpSize);
  out[grid.thread_rank()] = __shfl(var, src_lanes[block.thread_rank() % width], width);
}

template <typename T> class Shfl : public WarpTest<Shfl<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    const auto alloc_size = width_ * sizeof(uint8_t);
    LinearAllocGuard<uint8_t> src_lanes_dev(LinearAllocs::hipMalloc, alloc_size);
    src_lanes_.resize(width_);
    std::generate(src_lanes_.begin(), src_lanes_.end(),
                  [this] { return GenerateRandomInteger(0, static_cast<int>(2 * width_)); });

    HIP_CHECK(hipMemcpy(src_lanes_dev.ptr(), src_lanes_.data(), alloc_size, hipMemcpyHostToDevice));
    shfl<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                            src_lanes_dev.ptr(), width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto rank_in_partition = rank_in_block % width_;
      const int src_lane = src_lanes_[rank_in_partition] % width_;
      const int src_offset = src_lane - rank_in_partition;

      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      if (!active_mask.test(rank_in_warp) ||
          (!active_mask.test((rank_in_warp + src_offset) % this->warp_size_)) ||
          (rank_in_block + src_offset >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (i + src_offset) % this->warp_size_;
    });
  };

 private:
  std::vector<uint8_t> src_lanes_;
  int width_;
};

TEMPLATE_TEST_CASE("Unit_Warp_Functions_Shfl_Positive_Basic", "", int, unsigned int, long, unsigned long, long long,
                   unsigned long long, float, double) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  Shfl<TestType>().run();
}