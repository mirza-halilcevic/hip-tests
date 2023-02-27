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

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

static  uint64_t get_predicate_mask(unsigned int test_case, unsigned int warp_size) {
  uint64_t predicate_mask = 0;
  switch (test_case) {
    case 0:  // no thread
      predicate_mask = 0;
      break;
    case 1:  // 1st thread
      predicate_mask = 1;
      break;
    case 2:  // last thread
      predicate_mask = static_cast<uint64_t>(1) << (warp_size - 1);
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

namespace cg = cooperative_groups;

__global__ void kernel_ballot(uint64_t* const out, const uint64_t* const active_masks,
                              uint64_t predicate) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);

  out[grid.thread_rank()] = __ballot((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
}

class WarpBallot : public WarpTest<WarpBallot, uint64_t> {
 public:
  void launch_kernel(uint64_t* const arr_dev, const uint64_t* const active_masks) {
    auto test_case = GENERATE(range(0, 5));
    predicate_mask_ = get_predicate_mask(test_case, this->warp_size_);
    INFO("Predicate mask: " << predicate_mask_);
    kernel_ballot<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                     predicate_mask_);
  }

  void validate(const uint64_t* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<uint64_t> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto warp_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const auto block_rank = warp_idx / this->warps_in_block_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[warp_idx]);

      auto partition_size = this->warp_size_;
      if (warp_idx == this->warps_in_block_ * (block_rank + 1) - 1) {
        partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
      }

      if (!active_mask.test(rank_in_warp))
        return std::nullopt;
      else {
        auto active_predicate = get_active_predicate(predicate_mask_, partition_size);
        return (active_predicate & this->active_masks_[warp_idx]);
      }
    });
  }
 private:
  uint64_t predicate_mask_;
};

TEST_CASE("Unit_Warp_Ballot_Positive_Basic") {
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

__global__ void kernel_any(int* const out, const uint64_t* const active_masks,
                           uint64_t predicate) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);

  out[grid.thread_rank()] = __any((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
}

class WarpAny : public WarpTest<WarpAny, int> {
 public:
  void launch_kernel(int* const arr_dev, const uint64_t* const active_masks) {
    auto test_case = GENERATE(range(0, 5));
    predicate_mask_ = get_predicate_mask(test_case, this->warp_size_);
    INFO("Predicate mask: " << predicate_mask_);
    kernel_any<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                  predicate_mask_);
  }

  void validate(const int* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<int> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto warp_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const auto block_rank = warp_idx / this->warps_in_block_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[warp_idx]);

      auto partition_size = this->warp_size_;
      if (warp_idx == this->warps_in_block_ * (block_rank + 1) - 1) {
        partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
      }

      if (!active_mask.test(rank_in_warp))
        return std::nullopt;
      else {
        auto active_predicate = get_active_predicate(predicate_mask_, partition_size);
        return ((active_predicate & this->active_masks_[warp_idx]) != 0);
      }
    });
  }
 private:
  uint64_t predicate_mask_;
};

TEST_CASE("Unit_Warp_Vote_Any_Positive_Basic") {
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

__global__ void kernel_all(int* const out, const uint64_t* const active_masks,
                           uint64_t predicate) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);

  out[grid.thread_rank()] = __all((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
}

class WarpAll : public WarpTest<WarpAll, int> {
 public:
  void launch_kernel(int* const arr_dev, const uint64_t* const active_masks) {
    auto test_case = GENERATE(range(0, 5));
    predicate_mask_ = get_predicate_mask(test_case, this->warp_size_);
    INFO("Predicate mask: " << predicate_mask_);
    kernel_all<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks,
                                                                  predicate_mask_);
  }

  void validate(const int* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<int> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto warp_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const auto block_rank = warp_idx / this->warps_in_block_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[warp_idx]);

      auto partition_size = this->warp_size_;
      if (warp_idx == this->warps_in_block_ * (block_rank + 1) - 1) {
        partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
      }

      if (!active_mask.test(rank_in_warp))
        return std::nullopt;
      else {
        auto active_predicate = get_active_predicate(predicate_mask_, partition_size);
        return check_if_all(active_predicate, this->active_masks_[warp_idx], partition_size);
      }
    });
  }
 private:
  uint64_t predicate_mask_;
};

TEST_CASE("Unit_Warp_Vote_All_Positive_Basic") {
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