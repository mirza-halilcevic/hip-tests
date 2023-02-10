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

static uint64_t get_predicate_mask(unsigned int test_case) {
  uint64_t active_mask = 0;
  switch (test_case) {
    case 0:  // 1st thread
      active_mask = 1;
      break;
    case 1:  // last thread
      active_mask = static_cast<uint64_t>(1) << (kWarpSize - 1);
      break;
    case 2:  // all threads
      active_mask = 0xFFFFFFFFFFFFFFFF;
      break;
    case 3:  // every second thread
      active_mask = 0xAAAAAAAAAAAAAAAA;
      break;
    default:  // random
      static std::mt19937_64 mt(test_case);
      std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
      active_mask = dist(mt);
  }
  return active_mask;
}

static uint64_t get_active_predicate(uint64_t predicate, size_t partition_size) {
  uint64_t active_predicate = predicate;
  for (int i = partition_size; i < 64; i++) {
    active_predicate &= ~(static_cast<uint64_t>(1) << i);
  }
  return active_predicate;
}

static bool check_if_all(uint64_t predicate_mask, size_t partition_size) {

  for (int i = 0; i < partition_size; i++) {
    if (!(predicate_mask & (static_cast<uint64_t>(1) << i))) return false;
  }
  return true;
}

__global__ void  kernel_ballot(uint64_t* const out, uint64_t predicate, size_t warp_size) {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const auto warps_per_block = (block_size + warp_size - 1) / warp_size;
  const auto warp_id = thread_rank_in_block % warp_size; 
  const auto warp_rank = block_rank_in_grid * warps_per_block + thread_rank_in_block / warp_size;

  out[warp_rank] = __ballot((predicate & (static_cast<uint64_t>(1) << warp_id)));
}

TEST_CASE("Unit_Ballot_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpBallot) {
    HipTest::HIP_SKIP_TEST("Device doesn't support warp ballot!");
    return;
  }

  const auto warp_size = device_properties.warpSize;

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  CPUGrid grid(blocks, threads);

  auto test_case = GENERATE(range(0, 4));
  uint64_t predicate = get_predicate_mask(test_case);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;

  LinearAllocGuard<uint64_t> arr_dev(LinearAllocs::hipMalloc, warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> arr(LinearAllocs::hipHostMalloc, warps_in_grid * sizeof(uint64_t));

  kernel_ballot<<<blocks, threads>>>(arr_dev.ptr(), predicate, warp_size);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), warps_in_grid * sizeof(uint64_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < grid.block_count_; i++) {
    size_t partition_size = warp_size;
    auto active_predicate = get_active_predicate(predicate, partition_size);
    for (int j = 0; j < warps_in_block; j++) {
      if ( j == warps_in_block - 1) {
        partition_size = grid.threads_in_block_count_ - (warps_in_block - 1) * warp_size;
        active_predicate = get_active_predicate(predicate, partition_size);
      }
      if (arr.ptr()[i*warps_in_block + j] != active_predicate) {
        REQUIRE(arr.ptr()[i*warps_in_block + j] == active_predicate);
      }
    }
  }
}

__global__ void kernel_any(uint64_t* const out, uint64_t predicate, size_t warp_size) {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const auto warps_per_block = (block_size + warp_size - 1) / warp_size;
  const auto warp_id = thread_rank_in_block % warp_size; 
  const auto warp_rank = block_rank_in_grid * warps_per_block + thread_rank_in_block / warp_size;

  out[warp_rank] = __any((predicate & (static_cast<uint64_t>(1) << warp_id)));
}

TEST_CASE("Unit_Any_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpVote) {
    HipTest::HIP_SKIP_TEST("Device doesn't support warp vote!");
    return;
  }

  const auto warp_size = device_properties.warpSize;

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  CPUGrid grid(blocks, threads);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;

  auto test_case = GENERATE(range(0, 4));
  uint64_t predicate = get_predicate_mask(test_case);

  LinearAllocGuard<uint64_t> arr_dev(LinearAllocs::hipMalloc, warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> arr(LinearAllocs::hipHostMalloc, warps_in_grid * sizeof(uint64_t));

  kernel_any<<<blocks, threads>>>(arr_dev.ptr(), predicate, warp_size);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), warps_in_grid * sizeof(uint64_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  

  for (int i = 0; i < grid.block_count_; i++) {
    size_t partition_size = warp_size;
    auto active_predicate = get_active_predicate(predicate, partition_size);
    unsigned int expected = active_predicate != 0 ? 1 : 0;
    for (int j = 0; j < warps_in_block; j++) {
      if ( j == warps_in_block - 1) {
        partition_size = grid.threads_in_block_count_ - (warps_in_block - 1) * kWarpSize;
        active_predicate = get_active_predicate(predicate, partition_size);
        expected = active_predicate != 0 ? 1 : 0;
      }
      if (arr.ptr()[i*warps_in_block + j] != expected) {
        REQUIRE(arr.ptr()[i*warps_in_block + j] == expected);
      }
    }
  }
}

__global__ void kernel_all(uint64_t* const out, uint64_t predicate, size_t warp_size) {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const auto warps_per_block = (block_size + warp_size - 1) / warp_size;
  const auto warp_id = thread_rank_in_block % warp_size; 
  const auto warp_rank = block_rank_in_grid * warps_per_block + thread_rank_in_block / warp_size;

  out[warp_rank] = __all((predicate & (static_cast<uint64_t>(1) << warp_id)));
}

TEST_CASE("Unit_All_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpVote) {
    HipTest::HIP_SKIP_TEST("Device doesn't support warp vote!");
    return;
  }

  const auto warp_size = device_properties.warpSize;

  const auto blocks = GenerateBlockDimensionsForShuffle();
  const auto threads = GenerateThreadDimensionsForShuffle();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);

  CPUGrid grid(blocks, threads);

  const auto warps_in_block = (grid.threads_in_block_count_ + kWarpSize - 1) / kWarpSize;
  const auto warps_in_grid = warps_in_block * grid.block_count_;

  auto test_case = GENERATE(range(0, 4));
  uint64_t predicate = get_predicate_mask(test_case);

  LinearAllocGuard<uint64_t> arr_dev(LinearAllocs::hipMalloc, warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> arr(LinearAllocs::hipHostMalloc, warps_in_grid * sizeof(uint64_t));

  kernel_all<<<blocks, threads>>>(arr_dev.ptr(), predicate, warp_size);
  HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), warps_in_grid * sizeof(uint64_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  

  for (int i = 0; i < grid.block_count_; i++) {
    size_t partition_size = warp_size;
    auto active_predicate = get_active_predicate(predicate, partition_size);
    unsigned int expected =  check_if_all(active_predicate, partition_size) ? 1 : 0;
    for (int j = 0; j < warps_in_block; j++) {
      if ( j == warps_in_block - 1) {
        partition_size = grid.threads_in_block_count_ - (warps_in_block - 1) * warp_size;
        active_predicate = get_active_predicate(predicate, partition_size);
        expected =  check_if_all(active_predicate, partition_size) ? 1 : 0;
      }
      if (arr.ptr()[i*warps_in_block + j] != expected) {
        REQUIRE(arr.ptr()[i*warps_in_block + j] == expected);
      }
    }
  }
}