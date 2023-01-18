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
