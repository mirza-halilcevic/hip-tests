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

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

static uint64_t get_active_predicate(uint64_t predicate, size_t partition_size) {
  uint64_t active_predicate = predicate;
  for (int i = partition_size; i < 64; i++) {
    active_predicate &= ~(static_cast<uint64_t>(1) << i);
  }
  return active_predicate;
}

static bool check_if_all(uint64_t predicate_mask, uint64_t active_mask, size_t partition_size) {
  if (!(predicate_mask & active_mask)) return false;
  for (int i = 0; i < partition_size; i++) {
    if (active_mask & (static_cast<uint64_t>(1) << i)) {
      if (!(predicate_mask & (static_cast<uint64_t>(1) << i))) return false;
    }
  }
  return true;
}

template <typename Derived> class WarpVote {
 public:
  WarpVote() : warp_size_{get_warp_size()} {}

  void run() {
    const auto blocks = GenerateBlockDimensions();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensions();
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);

    auto test_case = GENERATE(range(0, 5));
    predicate_mask_ = get_predicate_mask(test_case);
    INFO("Predicate mask: " << predicate_mask_);

    warps_in_block_ = (grid_.threads_in_block_count_ + warp_size_ - 1) / warp_size_;
    warps_in_grid_ = warps_in_block_ * grid_.block_count_;
    const auto alloc_size = warps_in_grid_ * sizeof(uint64_t);

    LinearAllocGuard<uint64_t> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<uint64_t> arr(LinearAllocs::hipHostMalloc, alloc_size);
    HIP_CHECK(hipMemset(arr_dev.ptr(), 0, alloc_size));

    LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc, alloc_size);
    active_masks_.resize(warps_in_grid_);
    std::generate(active_masks_.begin(), active_masks_.end(),
                  [] { return GenerateRandomInteger(0ul, std::numeric_limits<uint64_t>().max()); });

    HIP_CHECK(
        hipMemcpy(active_masks_dev.ptr(), active_masks_.data(), alloc_size, hipMemcpyHostToDevice));
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

  uint64_t get_predicate_mask(unsigned int test_case) const {
    uint64_t predicate_mask = 0;
    switch (test_case) {
      case 0:  // no thread
        predicate_mask = 0;
        break;
      case 1:  // 1st thread
        predicate_mask = 1;
        break;
      case 2:  // last thread
        predicate_mask = static_cast<uint64_t>(1) << (warp_size_ - 1);
        break;
      case 3:  // all threads
        predicate_mask = 0xFFFFFFFFFFFFFFFF;
        break;
      default:  // random
        static std::mt19937_64 mt(test_case);
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
        predicate_mask = dist(mt);
    }
    return predicate_mask;
  }

  Derived& cast_to_derived() { return reinterpret_cast<Derived&>(*this); }

 protected:
  const int warp_size_;
  CPUGrid grid_;
  unsigned int warps_in_block_;
  unsigned int warps_in_grid_;
  std::vector<uint64_t> active_masks_;
  uint64_t predicate_mask_;
};

namespace cg = cooperative_groups;

__global__ void kernel_ballot(uint64_t* const out, const uint64_t* const active_masks,
                              uint64_t predicate) {
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  if (active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank())) {
    out[idx] = __ballot((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
  }
}

class WarpBallot : public WarpVote<WarpBallot> {
 public:
  void launch_kernel(uint64_t* const arr_dev, const uint64_t* const active_masks) {
    kernel_ballot<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                     this->predicate_mask_);
  }

  void validate(const uint64_t* const arr) {
    ArrayAllOf(arr, this->warps_in_grid_, [this](unsigned int i) -> uint64_t {
      const auto block_rank = i / this->warps_in_block_;
      auto active_predicate = get_active_predicate(this->predicate_mask_, this->warp_size_);
      if (i == this->warps_in_block_ * (block_rank + 1) - 1) {
        auto partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
        active_predicate = get_active_predicate(this->predicate_mask_, partition_size);
      }
      return (active_predicate & this->active_masks_[i]);
    });
  }
};

TEST_CASE("Unit_Ballot") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpBallot) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Ballot!");
    return;
  }

  WarpBallot().run();
}

__global__ void kernel_any(uint64_t* const out, const uint64_t* const active_masks,
                           uint64_t predicate) {
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  if (active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank())) {
    out[idx] = __any((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
  }
}

class WarpAny : public WarpVote<WarpAny> {
 public:
  void launch_kernel(uint64_t* const arr_dev, const uint64_t* const active_masks) {
    kernel_any<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                  this->predicate_mask_);
  }

  void validate(const uint64_t* const arr) {
    ArrayAllOf(arr, this->warps_in_grid_, [this](unsigned int i) -> uint64_t {
      const auto block_rank = i / this->warps_in_block_;
      auto active_predicate = get_active_predicate(this->predicate_mask_, this->warp_size_);
      if (i == this->warps_in_block_ * (block_rank + 1) - 1) {
        auto partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
        active_predicate = get_active_predicate(this->predicate_mask_, partition_size);
      }
      return ((active_predicate & this->active_masks_[i]) != 0);
    });
  }
};

TEST_CASE("Unit_Any") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpVote) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Vote!");
    return;
  }

  WarpAny().run();
}

__global__ void kernel_all(uint64_t* const out, const uint64_t* const active_masks,
                           uint64_t predicate) {
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  if (active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank())) {
    out[idx] = __all((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
  }
}

class WarpAll : public WarpVote<WarpAll> {
 public:
  void launch_kernel(uint64_t* const arr_dev, const uint64_t* const active_masks) {
    kernel_all<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                  this->predicate_mask_);
  }

  void validate(const uint64_t* const arr) {
    ArrayAllOf(arr, this->warps_in_grid_, [this](unsigned int i) -> uint64_t {
      const auto block_rank = i / this->warps_in_block_;
      auto partition_size = this->warp_size_;
      auto active_predicate = get_active_predicate(this->predicate_mask_, partition_size);
      if (i == this->warps_in_block_ * (block_rank + 1) - 1) {
        partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
        active_predicate = get_active_predicate(this->predicate_mask_, partition_size);
      }
      return check_if_all(active_predicate, this->active_masks_[i], partition_size);
    });
  }
};

TEST_CASE("Unit_All") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpVote) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Vote!");
    return;
  }
  WarpAll().run();
}