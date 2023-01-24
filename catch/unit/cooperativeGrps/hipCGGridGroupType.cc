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
#include <hip/hip_cooperative_groups.h>

#include <resource_guards.hh>
#include <utils.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

namespace cg = cooperative_groups;

static __global__ void grid_group_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::this_grid().size();
}

static __global__ void grid_group_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::this_grid().thread_rank();
}

static __global__ void grid_group_is_valid_getter(bool* is_valid_flags) {
  is_valid_flags[thread_rank_in_grid()] = cg::this_grid().is_valid();
}

static __global__ void grid_group_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_grid());
}

static __global__ void grid_group_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_grid());
}

static __global__ void
sync_kernel(unsigned int *atomic_val, unsigned int *array,
            unsigned int loops) {
  cg::grid_group grid = cg::this_grid();
  unsigned rank = grid.thread_rank();

  int offset = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the sync below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets to it.
    // As such, without the sync, the last array entry will eventually
    // contain a very large value, defined by however many times the other
    // wavefronts make it through this loop.
    // If the sync works, then it will likely contain some number
    // near "total number of blocks". It will be the last wavefront to
    // reach the atomicInc, but everyone will have only hit the atomic once.
    if (rank == (grid.size() - 1)) {
      long long time_diff = 0;
      long long last_clock = clock64();
      do {
        long long cur_clock = clock64();
        if (cur_clock > last_clock) {
          time_diff += (cur_clock - last_clock);
        }
        // If it rolls over, we don't know how much to add to catch up.
        // So just ignore those slipped cycles.
        last_clock = cur_clock;
      } while(time_diff < 1000000);
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1) {
      array[offset] = atomicInc(&atomic_val[0], UINT_MAX);
    }
    grid.sync();
    offset += gridDim.x * gridDim.y * gridDim.z;
  }
}

TEST_CASE("Unit_Grid_Group_Getters_Positive_Basic") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);

  {
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                                grid.thread_count_ * sizeof(unsigned int));
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                            grid.thread_count_ * sizeof(unsigned int));

    // Launch Kernel
    unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
    void *params[1];
    params[0] = &uint_arr_dev_ptr;

    HIP_CHECK(hipLaunchCooperativeKernel(grid_group_size_getter, blocks, threads,
                               params, 0, 0));

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipLaunchCooperativeKernel(grid_group_thread_rank_getter, blocks, threads,
                             params, 0, 0));

    // Verify grid_group.size() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [size = grid.thread_count_](uint32_t) { return size; });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify grid_group.thread_rank() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [](uint32_t i) { return i; });
  }

  {
    LinearAllocGuard<bool> bool_arr_dev(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(bool));
    LinearAllocGuard<bool> bool_arr(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(bool));

    bool* bool_arr_dev_ptr = bool_arr_dev.ptr();
    void *params[1];
    params[0] = &bool_arr_dev_ptr;
    HIP_CHECK(hipLaunchCooperativeKernel(grid_group_is_valid_getter, blocks, threads,
                               params, 0, 0));

    HIP_CHECK(hipMemcpy(bool_arr.ptr(), bool_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*bool_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify grid_group.is_valid() values
    ArrayAllOf(bool_arr.ptr(), grid.thread_count_,
               [](uint32_t i) { return 1; });
  }
}

TEST_CASE("Unit_Grid_Group_Getters_Positive_Non_Member_Functions") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  // Launch Kernel
  unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
  void *params[1];
  params[0] = &uint_arr_dev_ptr;

  HIP_CHECK(hipLaunchCooperativeKernel(grid_group_non_member_size_getter, blocks, threads,
                              params, 0, 0));

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipLaunchCooperativeKernel(grid_group_non_member_thread_rank_getter, blocks, threads,
                            params, 0, 0));

  // Verify grid_group.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
            [size = grid.thread_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                    grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify grid_group.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
              [](uint32_t i) { return i; });
}

TEST_CASE("Unit_Grid_Group_Positive_Sync") {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  auto loops = GENERATE(2, 4, 8, 16);
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);
  unsigned int array_len = grid.block_count_ * loops;

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              array_len * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          array_len * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> atomic_val(LinearAllocs::hipMalloc,
                                           sizeof(unsigned int));
  HIP_CHECK(hipMemset(atomic_val.ptr(), 0, sizeof(unsigned int)));

  // Launch Kernel
  unsigned int* uint_arr_dev_ptr = uint_arr_dev.ptr();
  unsigned int* atomic_val_ptr = atomic_val.ptr();
  void *params[3];
  params[0] = reinterpret_cast<void*>(&atomic_val_ptr);
  params[1] = reinterpret_cast<void*>(&uint_arr_dev_ptr);
  params[2] = reinterpret_cast<void*>(&loops);

  HIP_CHECK(hipLaunchCooperativeKernel(sync_kernel, blocks, threads, params, 0, 0));

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        array_len * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));

  HIP_CHECK(hipDeviceSynchronize());

  // Verify host buffer values
  unsigned int max_in_this_loop = 0;
  for (unsigned int i = 0; i < loops; i++) {
    max_in_this_loop += grid.block_count_;
    unsigned int j = 0;
    for (j = 0; j < grid.block_count_ - 1; j++) {
      REQUIRE(uint_arr.ptr()[i*grid.block_count_+j] < max_in_this_loop);
    }
    REQUIRE(uint_arr.ptr()[i*grid.block_count_+ j] == max_in_this_loop - 1);
  }
}
