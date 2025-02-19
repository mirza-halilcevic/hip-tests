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

#include "warp_vote_common.hh"

#include <bitset>

/**
 * @addtogroup any any
 * @{
 * @ingroup DeviceLanguageTest
 * `int __any(int predicate)` -
 * Contains unit test for warp vote any function
 */

namespace cg = cooperative_groups;


__global__ void kernel_any(int* const out, const uint64_t* const active_masks, uint64_t predicate) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  const auto warp = cg::tiled_partition(cg::this_thread_block(), warpSize);

  out[grid.thread_rank()] = __any((predicate & (static_cast<uint64_t>(1) << warp.thread_rank())));
}

class WarpAny : public WarpVoteTest<WarpAny, int> {
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
      // If the number of threads in a block is not a multiple of warp size, the
      // last warp will have inactive threads and partition size must be recalculated
      if (warp_idx == this->warps_in_block_ * (block_rank + 1) - 1) {
        partition_size =
            this->grid_.threads_in_block_count_ - (this->warps_in_block_ - 1) * this->warp_size_;
      }

      if (!active_mask.test(rank_in_warp))
        return std::nullopt;
      else {
        // Active predicate mask must be calculated as partition can be smaller than warp_size
        auto active_predicate = get_active_predicate(predicate_mask_, partition_size);
        return ((active_predicate & this->active_masks_[warp_idx]) != 0);
      }
    });
  }

 private:
  uint64_t predicate_mask_;
};

/**
 * Test Description
 * ------------------------
 *  - Validates the warp vote any function behavior. Threads are deactivated based on the passed
 * active mask. The predicate for each thread is determined according to the generated predicate
 * mask.
 * Test source
 * ------------------------
 *  - unit/warp/warp_any.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports warp vote
 */
TEST_CASE("Unit_Warp_Vote_Any_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpVote) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Vote!");
    return;
  }

  SECTION("Warp Vote Any with specified active mask") {
    WarpAny().run(false);
  }

  SECTION("Warp Vote Any with random active mask") {
    WarpAny().run(true);
  }
}