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

namespace cg = cooperative_groups;

__device__ bool deactivate_thread(const uint64_t* const active_masks) {
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  return !(active_masks[idx] & (1u << warp.thread_rank()));
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename Derived, typename T> class Foo {
 public:
  Foo() : warp_size_{get_warp_size()} {}

  void run() {
    const auto blocks = GenerateBlockDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);
    width_ = generate_width();
    INFO("Width: " << width_);

    const auto alloc_size = grid_.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    warps_in_block_ = (grid_.threads_in_block_count_ + warp_size_ - 1) / warp_size_;
    const auto warps_in_grid = warps_in_block_ * grid_.block_count_;
    LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                                warps_in_grid * sizeof(uint64_t));
    active_masks_.resize(warps_in_grid);
    std::generate(active_masks_.begin(), active_masks_.end(),
                [] { return GenerateRandomInteger(0u, std::numeric_limits<uint32_t>().max()); });

    HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks_.data(),
                        warps_in_grid * sizeof(uint64_t), hipMemcpyHostToDevice));
    cast_to_derived().launch_kernel(arr_dev.ptr(), active_masks_dev.ptr());
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
  unsigned int warps_in_block_;
  int width_;
  std::vector<uint64_t> active_masks_;
};

namespace cg = cooperative_groups;

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

template <typename T> class ShflUp : public Foo<ShflUp<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    delta_ = GENERATE_COPY(range(0, this->width_));
    INFO("Delta: " << delta_);
    shfl_up<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, delta_,
                                                               this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % this->width_ - delta_;
      if (!active_mask.test(rank_in_warp) ||
          (target >= 0 && !active_mask.test(rank_in_warp - delta_))) {
        return std::nullopt;
      }

      return (target < 0 ? i : i - delta_) % this->warp_size_;
    });
  };

 private:
  unsigned int delta_;
};

TEMPLATE_TEST_CASE("ShflUp", "", int) { ShflUp<TestType>().run(); }


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

template <typename T> class ShflDown : public Foo<ShflDown<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    delta_ = GENERATE_COPY(range(0, this->width_));
    INFO("Delta: " << delta_);
    shfl_down<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, delta_,
                                                                 this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const int rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % this->width_ + delta_;

      if (!active_mask.test(rank_in_warp) ||
          (target < this->width_ && !active_mask.test(rank_in_warp + delta_)) ||
          (rank_in_block + delta_ >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (target >= this->width_ ? i : i + delta_) % this->warp_size_;
    });
  };

 private:
  unsigned int delta_;
};

TEMPLATE_TEST_CASE("ShflDown", "", int, float) { ShflDown<TestType>().run(); }


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

template <typename T> class ShflXOR : public Foo<ShflXOR<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    lane_mask_ = GENERATE_COPY(range(0, this->warp_size_));
    INFO("Lane mask: " << lane_mask_);
    shfl_xor<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, lane_mask_,
                                                                this->width_);
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

      const auto target_partition = warp_target / this->width_;
      const auto partition_rank = rank_in_warp / this->width_;
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
};

TEMPLATE_TEST_CASE("ShflXOR", "", int, float) { ShflXOR<TestType>().run(); }


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

template <typename T> class Shfl : public Foo<Shfl<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    const auto alloc_size = this->width_ * sizeof(uint8_t);
    LinearAllocGuard<uint8_t> src_lanes_dev(LinearAllocs::hipMalloc, alloc_size);
    src_lanes_.resize(this->width_);
    std::generate(src_lanes_.begin(), src_lanes_.end(),
                [this] { return GenerateRandomInteger(0, static_cast<int>(2 * this->width_)); });

    HIP_CHECK(hipMemcpy(src_lanes_dev.ptr(), src_lanes_.data(), alloc_size, hipMemcpyHostToDevice));
    shfl<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                            src_lanes_dev.ptr(), this->width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto rank_in_partition = rank_in_block % this->width_;
      const int src_lane = src_lanes_[rank_in_partition] % this->width_;
      const int src_offset = src_lane - rank_in_partition;

      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      if (!active_mask.test(rank_in_warp) || (!active_mask.test((rank_in_warp + src_offset) % this->warp_size_)) ||
          (rank_in_block + src_offset >= this->grid_.threads_in_block_count_)) {
        return std::nullopt;
      }

      return (i + src_offset) % this->warp_size_;
    });
  };

 private:
  std::vector<uint8_t> src_lanes_;
};

TEMPLATE_TEST_CASE("Shfl", "", int, float) { Shfl<TestType>().run(); }