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

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_size_getter(unsigned int* sizes) {
  const BaseType group = cg::this_multi_grid();
  sizes[thread_rank_in_grid()] = group.size();
}

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_thread_rank_getter(unsigned int* thread_ranks) {
  const BaseType group = cg::this_multi_grid();
  thread_ranks[thread_rank_in_grid()] = group.thread_rank();
}

template <typename BaseType = cg::multi_grid_group>
static __global__ void multi_grid_group_is_valid_getter(unsigned int* is_valid_flags) {
  const BaseType group = cg::this_multi_grid();
  is_valid_flags[thread_rank_in_grid()] = cg::this_multi_grid().is_valid();
}

static __global__ void multi_grid_group_num_grids_getter(unsigned int* num_grids) {
  num_grids[thread_rank_in_grid()] = cg::this_multi_grid().num_grids();
}

static __global__ void multi_grid_group_grid_rank_getter(unsigned int* grid_ranks) {
  grid_ranks[thread_rank_in_grid()] = cg::this_multi_grid().grid_rank();
}

static __global__ void multi_grid_group_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_multi_grid());
}

static __global__ void multi_grid_group_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_multi_grid());
}

static __global__ void sync_kernel(unsigned int* atomic_val, unsigned int* global_array,
                                   unsigned int* array, uint32_t loops) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cooperative_groups::multi_grid_group mgrid = cooperative_groups::this_multi_grid();
  unsigned rank = grid.thread_rank();
  unsigned global_rank = mgrid.thread_rank();

  int offset = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  for (int i = 0; i < loops; i++) {
    // Make the last thread run way behind everyone else.
    // If the grid barrier below fails, then the other threads may hit the
    // atomicInc instruction many times before the last thread ever gets
    // to it.
    // As such, without the barrier, the last array entry will eventually
    // contain a very large value, defined by however many times the other
    // wavefronts make it through this loop.
    // If the barrier works, then it will likely contain some number
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
      } while (time_diff < 1000000);
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 &&
        threadIdx.z == blockDim.z - 1) {
      array[offset] = atomicInc(atomic_val, UINT_MAX);
    }
    grid.sync();

    // Make the last thread in the entire multi-grid run way behind
    // everyone else.
    // If the mgrid barrier below fails, then the two global_array entries
    // will end up being out of sync, because the intermingling of adds
    // and multiplies will not be aligned between to the two GPUs.
    if (global_rank == (mgrid.size() - 1)) {
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
      } while (time_diff < 1000000);
    }
    // During even iterations, add into your own array entry
    // During odd iterations, add into next array entry
    unsigned grid_rank = mgrid.grid_rank();
    unsigned inter_gpu_offset = (grid_rank + 1) % mgrid.num_grids();
    if (rank == (grid.size() - 1)) {
      if (i % 2 == 0) {
        global_array[grid_rank] += 2;
      } else {
        global_array[inter_gpu_offset] *= 2;
      }
    }
    mgrid.sync();
    offset += gridDim.x * gridDim.y * gridDim.z;
  }
}

TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Basic") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, MaxGPUs);
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);

  // Calculate total thread count and local grid ranks
  unsigned int multi_grid_thread_count = 0;
  unsigned int multi_grid_grid_rank_0[MaxGPUs];
  multi_grid_grid_rank_0[0] = 0;
  for (int i = 0; i < num_devices; i++) {
    if (i > 0) {
      multi_grid_grid_rank_0[i] = multi_grid_thread_count;
    }
    multi_grid_thread_count += grid.thread_count_;
  }

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  unsigned int* uint_arr_dev_ptr[num_devices];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  hipLaunchParams launchParamsList[num_devices];
  void* args[num_devices];
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_size_getter<cg::multi_grid_group>);
    launchParamsList[i].gridDim = blocks;
    launchParamsList[i].blockDim = threads;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_thread_rank_getter<cg::multi_grid_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [size = multi_grid_thread_count](uint32_t) { return size; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_grid_rank_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.thread_rank() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [rank_0 = multi_grid_grid_rank_0[i]](uint32_t j) { return rank_0 + j; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_num_grids_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.grid_rank() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_, [rank = i](uint32_t) { return rank; });

    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_is_valid_getter<cg::multi_grid_group>);
  }

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.num_grids() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [num = num_devices](uint32_t) { return num; });

    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify multi_grid_group.is_valid() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [num = num_devices](uint32_t) { return num; });
  }
}

TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Base_Type") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, MaxGPUs);

  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);

  // Calculate total thread count and local grid ranks
  unsigned int multi_grid_thread_count = 0;
  unsigned int multi_grid_grid_rank_0[MaxGPUs];
  multi_grid_grid_rank_0[0] = 0;
  for (int i = 0; i < num_devices; i++) {
    if (i > 0) {
      multi_grid_grid_rank_0[i] = multi_grid_thread_count;
    }
    multi_grid_thread_count += grid.thread_count_;
  }

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  unsigned int* uint_arr_dev_ptr[num_devices];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  hipLaunchParams launchParamsList[num_devices];
  void* args[num_devices];
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_size_getter<cg::thread_group>);
    launchParamsList[i].gridDim = blocks;
    launchParamsList[i].blockDim = threads;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_thread_rank_getter<cg::thread_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [size = multi_grid_thread_count](uint32_t) { return size; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_is_valid_getter<cg::thread_group>);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.thread_rank() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [rank_0 = multi_grid_grid_rank_0[i]](uint32_t j) { return rank_0 + j; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify multi_grid_group.is_valid() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [](uint32_t j) { return 1; });
  }
}

TEST_CASE("Unit_Multi_Grid_Group_Getters_Positive_Non_Member_Functions") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, MaxGPUs);

  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);

  // Calculate total thread count and local grid ranks
  unsigned int multi_grid_thread_count = 0;
  unsigned int multi_grid_grid_rank_0[MaxGPUs];
  multi_grid_grid_rank_0[0] = 0;
  for (int i = 0; i < num_devices; i++) {
    if (i > 0) {
      multi_grid_grid_rank_0[i] = multi_grid_thread_count;
    }
    multi_grid_thread_count += grid.thread_count_;
  }

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  unsigned int* uint_arr_dev_ptr[num_devices];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(unsigned int));
  }

  // Launch Kernel
  hipLaunchParams launchParamsList[num_devices];
  void* args[num_devices];
  for (int i = 0; i < num_devices; i++) {
    args[i] = &uint_arr_dev_ptr[i];

    launchParamsList[i].func = reinterpret_cast<void*>(multi_grid_group_non_member_size_getter);
    launchParamsList[i].gridDim = blocks;
    launchParamsList[i].blockDim = threads;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = streams[i].stream();
    launchParamsList[i].args = &args[i];
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    launchParamsList[i].func =
        reinterpret_cast<void*>(multi_grid_group_non_member_thread_rank_getter);
  }
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, num_devices, 0));

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    // Verify multi_grid_group.size() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [size = multi_grid_thread_count](uint32_t) { return size; });
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(),
                        grid.thread_count_ * sizeof(*uint_arr[i].ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    // Verify multi_grid_group.thread_rank() values
    ArrayAllOf(uint_arr[i].ptr(), grid.thread_count_,
               [rank_0 = multi_grid_grid_rank_0[i]](uint32_t j) { return rank_0 + j; });
  }
}

TEST_CASE("Unit_Multi_Grid_Group_Positive_Sync") {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  num_devices = min(num_devices, MaxGPUs);

  hipDeviceProp_t device_properties[num_devices];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipGetDeviceProperties(&device_properties[i], i));
    if (!device_properties[i].cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
  }
  auto loops = GENERATE(2, 4, 8, 16);
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(2, 2, 2));

  const CPUGrid grid(blocks, threads);
  unsigned int array_len = grid.block_count_ * loops;

  std::vector<StreamGuard> streams;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr_dev;
  std::vector<LinearAllocGuard<unsigned int>> uint_arr;
  std::vector<LinearAllocGuard<unsigned int>> atomic_val;
  unsigned int* uint_arr_dev_ptr[num_devices];
  unsigned int* atomic_val_ptr[num_devices];
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    streams.emplace_back(Streams::created);

    // Allocate grid sync arrays
    uint_arr_dev.emplace_back(LinearAllocs::hipMalloc, array_len * sizeof(unsigned int));
    uint_arr_dev_ptr[i] = uint_arr_dev[i].ptr();
    uint_arr.emplace_back(LinearAllocs::hipHostMalloc, array_len * sizeof(unsigned int));

    atomic_val.emplace_back(LinearAllocs::hipMalloc, sizeof(unsigned int));
    HIP_CHECK(hipMemset(atomic_val[i].ptr(), 0, sizeof(unsigned int)));
    atomic_val_ptr[i] = atomic_val[i].ptr();
  }
  // Allocate multi_grid sync array
  LinearAllocGuard<unsigned int> global_arr(LinearAllocs::hipHostMalloc,
                                            num_devices * sizeof(unsigned int));
  HIP_CHECK(hipMemset(global_arr.ptr(), 0, num_devices * sizeof(unsigned int)));
  unsigned int* global_arr_ptr = global_arr.ptr();

  void* dev_params[num_devices][4];
  hipLaunchParams md_params[num_devices];
  for (int i = 0; i < num_devices; i++) {
    dev_params[i][0] = reinterpret_cast<void*>(&atomic_val_ptr[i]);
    dev_params[i][1] = reinterpret_cast<void*>(&global_arr_ptr);
    dev_params[i][2] = reinterpret_cast<void*>(&uint_arr_dev_ptr[i]);
    dev_params[i][3] = reinterpret_cast<void*>(&loops);

    md_params[i].func = reinterpret_cast<void*>(sync_kernel);
    md_params[i].gridDim = blocks;
    md_params[i].blockDim = threads;
    md_params[i].sharedMem = 0;
    md_params[i].stream = streams[i].stream();
    md_params[i].args = dev_params[i];
  }

  // Launch Kernel
  HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(md_params, num_devices, 0));
  HIP_CHECK(hipDeviceSynchronize());

  // Read back the grid sync buffer to host
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipMemcpy(uint_arr[i].ptr(), uint_arr_dev[i].ptr(), array_len * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipDeviceSynchronize());

  // Verify grid sync host array values
  for (int i = 0; i < num_devices; i++) {
    unsigned int max_in_this_loop = 0;
    for (unsigned int j = 0; j < loops; j++) {
      max_in_this_loop += grid.block_count_;
      unsigned int k = 0;
      for (k = 0; k < grid.block_count_ - 1; k++) {
        REQUIRE(uint_arr[i].ptr()[j * grid.block_count_ + k] < max_in_this_loop);
      }
      REQUIRE(uint_arr[i].ptr()[j * grid.block_count_ + k] == max_in_this_loop - 1);
    }
  }

  // Verify multi_grid sync array values
  const auto f = [loops](unsigned int i) -> unsigned int {
    unsigned int desired_val = 0;
    for (int j = 0; j < loops; j++) {
      if (j % 2 == 0) {
        desired_val += 2;
      } else {
        desired_val *= 2;
      }
    }
    return desired_val;
  };
  ArrayAllOf(global_arr.ptr(), num_devices, f);
}
