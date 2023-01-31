/*
Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

// Test Description:
/* This test implements sum reduction kernel, first with each threads own rank
   as input and comparing the sum with expected sum output derieved from n(n-1)/2
   formula. The second part, partitions this parent group into child subgroups
   a.k.a tiles using using tiled_partition() collective operation. This can be called
   with a static tile size, passed in templated non-type variable-tiled_partition<tileSz>,
   or in runtime as tiled_partition(thread_group parent, tileSz). This test covers both these
   cases.
   This test tests functionality of cg group partitioning, (static and dynamic) and its respective
   API's size(), thread_rank(), and sync().
*/
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <resource_guards.hh>
#include <utils.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

namespace cg = cooperative_groups;

static constexpr size_t kWarpSize = 32;

template <typename BaseType = cg::coalesced_group>
static __global__ void coalesced_group_size_getter(unsigned int* sizes, uint64_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    BaseType active = cg::coalesced_threads();
    sizes[thread_rank_in_grid()] = active.size();
  } 
}

template <typename BaseType = cg::coalesced_group>
static __global__ void coalesced_group_thread_rank_getter(unsigned int* thread_ranks, uint64_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    BaseType active = cg::coalesced_threads();
    thread_ranks[thread_rank_in_grid()] = active.thread_rank();
  } 
}

#if HT_AMD
static __global__ void coalesced_group_is_valid_getter(unsigned int* is_valid_flags, uint64_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::thread_group active = cg::coalesced_threads();
    is_valid_flags[thread_rank_in_grid()] = cg::is_valid(active);
  } 
}
#endif

static __global__ void coalesced_group_non_member_size_getter(unsigned int* sizes, uint64_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    sizes[thread_rank_in_grid()] = cg::group_size(active);
  } 
}

static __global__ void coalesced_group_non_member_thread_rank_getter(unsigned int* thread_ranks, uint64_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    thread_ranks[thread_rank_in_grid()] = cg::thread_rank(active);
  } 
}

unsigned int get_active_thread_count(uint32_t active_mask, unsigned int partition_size) {
  unsigned int active_thread_count = 0;
  for (int i = 0; i < partition_size; i ++) {
    if (active_mask & (1 << i)) active_thread_count++;
  }
  return active_thread_count;
}

TEST_CASE("Unit_Coalesced_Group_Getters_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8), dim3(64, 2, 1));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  uint64_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  unsigned int partition_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + kWarpSize - 1) / kWarpSize;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * kWarpSize) {
      partition_size = grid.threads_in_block_count_ - (partitions_in_block - 1) * kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      partition_size = kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    }
    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                    grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

TEST_CASE("Unit_Coalesced_Group_Getters_Positive_Base_Type") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8), dim3(64, 2, 1));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  uint64_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);

  const CPUGrid grid(blocks, threads);
 
  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_size_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_thread_rank_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  unsigned int partition_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + kWarpSize - 1) / kWarpSize;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * kWarpSize) {
      partition_size = grid.threads_in_block_count_ - (partitions_in_block - 1) * kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      partition_size = kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    }
    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                    grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

TEST_CASE("Unit_Coalesced_Group_Getters_Positive_Non_Member_Functions") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8), dim3(64, 2, 1));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  uint64_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));

  // Launch Kernel
  coalesced_group_non_member_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(uint_arr_dev.ptr(), 0, grid.thread_count_ * sizeof(unsigned int)));
  HIP_CHECK(hipDeviceSynchronize());
  coalesced_group_non_member_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr(), active_mask);

  // Verify coalesced_group.size() values
  unsigned int coalesced_size = 0;
  unsigned int partition_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + kWarpSize - 1) / kWarpSize;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    // If the number of threads in a block is not a multiple of warp size, the
    // last warp will have inactive threads and coalesced group size must be recalculated
    if (rank_in_block == (partitions_in_block - 1) * kWarpSize) {
      partition_size = grid.threads_in_block_count_ - (partitions_in_block - 1) * kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    } else if (rank_in_block == 0) {
      partition_size = kWarpSize;
      coalesced_size = get_active_thread_count(active_mask, partition_size);
    }
    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_size) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_size);
      }
    }
  }

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                    grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify coalesced_group.thread_rank() values
  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % kWarpSize;
    if (rank_in_partition == 0) coalesced_rank = 0;

    if (active_mask & (1 << rank_in_partition)) {
      if (uint_arr.ptr()[i] != coalesced_rank) {
        REQUIRE(uint_arr.ptr()[i] == coalesced_rank);
      }
      coalesced_rank++;
    }
  }
}

template <typename T>
__global__ void coalesced_group_shfl_up(T* const out, const unsigned int delta, const uint32_t active_mask) {
  const auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl_up(var, delta);
  }
}

template <typename T> void CoalescedGroupShflUpTestImpl() {
  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  uint32_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);
  unsigned int active_thread_count = get_active_thread_count(active_mask, 32);

  auto delta = GENERATE(range(static_cast<size_t>(0), static_cast<size_t>(32)));
  delta = delta % active_thread_count;
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  coalesced_group_shfl_up<T><<<blocks, threads>>>(arr_dev.ptr(), delta, active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  unsigned int coalesced_rank = 0;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % 32;
    if (rank_in_partition == 0) coalesced_rank = 0;
    if (active_mask & (1 << rank_in_partition)) {
      int target = coalesced_rank - delta;
      target = target < 0 ? coalesced_rank : target;
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}

TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Up_Positive_Basic", "", int) {
  CoalescedGroupShflUpTestImpl<TestType>();
}

template <typename T>
__global__ void coalesced_group_shfl_down(T* const out, const unsigned int delta, const uint32_t active_mask) {
  const auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl_down(var, delta);
  }
}

template <typename T> void CoalescedGroupShflDownTest() {
  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  uint32_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);
  unsigned int active_thread_count = get_active_thread_count(active_mask, 32);

  auto delta = GENERATE(range(static_cast<size_t>(0), static_cast<size_t>(32)));
  delta = delta % active_thread_count;
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  coalesced_group_shfl_down<T><<<blocks, threads>>>(arr_dev.ptr(), delta, active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  unsigned int coalesced_rank = 0;
  unsigned int group_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + 31) / 32;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % 32;
    if (rank_in_partition == 0) coalesced_rank = 0;
    if (rank_in_block < (partitions_in_block - 1) * 32) {
      group_size = get_active_thread_count(active_mask, 32);
    }
    else {
      const auto tail_size = grid.threads_in_block_count_ - (partitions_in_block - 1) * 32;
      group_size = get_active_thread_count(active_mask, tail_size);
    }
    if (active_mask & (1 << rank_in_partition)) {
      int target = coalesced_rank + delta;
      target = target < group_size ? target : coalesced_rank;
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}

TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Down_Positive_Basic", "", int) {
  CoalescedGroupShflDownTest<TestType>();
}

template <typename T>
__global__ void coalesced_group_shfl(T* const out, uint8_t* target_lanes, const uint32_t active_mask) {
  const auto tile = cg::tiled_partition<32>(cg::this_thread_block());
  if (active_mask & (1 << tile.thread_rank())) {
    cg::coalesced_group active = cg::coalesced_threads();
    T var = static_cast<T>(active.thread_rank());
    out[thread_rank_in_grid()] = active.shfl(var, target_lanes[active.thread_rank()]);;
  }
}

template <typename T> void CoalescedGroupShflTest() {
  auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8));
  auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
  CPUGrid grid(blocks, threads);

  uint32_t active_mask = GENERATE(0x1, 0xFFFFFFFF, 0xAAAAAAAA, 0x24924924, 0x88888888, 0x21084210, 0x20080200, 0x80000000);
  unsigned int active_thread_count = get_active_thread_count(active_mask, 32);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

  LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc,
                                             active_thread_count * sizeof(uint8_t));
  LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc,
                                         active_thread_count * sizeof(uint8_t));
  // Generate a couple different combinations for target lanes
  for (auto i = 0u; i < active_thread_count; ++i) {
    target_lanes.ptr()[i] = active_thread_count - 1 - i;
  }

  HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(), active_thread_count * sizeof(uint8_t),
                      hipMemcpyHostToDevice));
  coalesced_group_shfl<T><<<blocks, threads>>>(arr_dev.ptr(), target_lanes_dev.ptr(), active_mask);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());


  unsigned int coalesced_rank = 0;
  unsigned int group_size = 0;
  const auto partitions_in_block = (grid.threads_in_block_count_ + 31) / 32;
  for (int i = 0; i < grid.thread_count_; i++) {
    const auto rank_in_block = grid.thread_rank_in_block(i).value();
    const int rank_in_partition = rank_in_block % 32;
    if (rank_in_partition == 0) coalesced_rank = 0;
    if (rank_in_block < (partitions_in_block - 1) * 32) {
      group_size = get_active_thread_count(active_mask, 32);
    }
    else {
      const auto tail_size = grid.threads_in_block_count_ - (partitions_in_block - 1) * 32;
      group_size = get_active_thread_count(active_mask, tail_size);
    }
    if (active_mask & (1 << rank_in_partition)) {
      auto target = target_lanes.ptr()[coalesced_rank];
      if (target >= group_size) target = 0;
      if (arr.ptr()[i] != target) {
        REQUIRE(arr.ptr()[i] == target);
      }
      coalesced_rank++;
    }
  }
}

TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Shfl_Positive_Basic", "", int) {
  CoalescedGroupShflTest<TestType>();
}
