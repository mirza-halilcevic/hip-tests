/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <optional>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <resource_guards.hh>
#include <utils.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

namespace cg = cooperative_groups;

__device__ unsigned int thread_rank_in_grid() {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  return block_rank_in_grid * block_size + thread_rank_in_block;
}

template <typename BaseType = cg::thread_block>
static __global__ void thread_block_size_getter(unsigned int* sizes) {
  const BaseType group = cg::this_thread_block();
  sizes[thread_rank_in_grid()] = group.size();
}

template <typename BaseType = cg::thread_block>
static __global__ void thread_block_thread_rank_getter(unsigned int* thread_ranks) {
  const BaseType group = cg::this_thread_block();
  thread_ranks[thread_rank_in_grid()] = group.thread_rank();
}

static __global__ void thread_block_group_indices_getter(dim3* group_indices) {
  group_indices[thread_rank_in_grid()] = cg::this_thread_block().group_index();
}

static __global__ void thread_block_thread_indices_getter(dim3* thread_indices) {
  thread_indices[thread_rank_in_grid()] = cg::this_thread_block().thread_index();
}

static __global__ void thread_block_group_dims_getter(dim3* group_dims) {
  group_dims[thread_rank_in_grid()] = cg::this_thread_block().group_dim();
}

static __global__ void thread_block_non_member_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::group_size(cg::this_thread_block());
}

static __global__ void thread_block_non_member_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::thread_rank(cg::this_thread_block());
}

TEST_CASE("Unit_Thread_Block_Getters_Positive_Basic") {
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(10, 10, 10));

  const CPUGrid grid(blocks, threads);

  {
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                                grid.thread_count_ * sizeof(unsigned int));
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                            grid.thread_count_ * sizeof(unsigned int));

    thread_block_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr());

    // Verify thread_block.size() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [size = grid.threads_in_block_count_](uint32_t) { return size; });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify thread_block.thread_rank() values
    ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
  }

  {
    LinearAllocGuard<dim3> dim3_arr_dev(LinearAllocs::hipMalloc, grid.thread_count_ * sizeof(dim3));
    LinearAllocGuard<dim3> dim3_arr(LinearAllocs::hipHostMalloc, grid.thread_count_ * sizeof(dim3));

    thread_block_group_indices_getter<<<blocks, threads>>>(dim3_arr_dev.ptr());
    HIP_CHECK(hipMemcpy(dim3_arr.ptr(), dim3_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*dim3_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_thread_indices_getter<<<blocks, threads>>>(dim3_arr_dev.ptr());

    // Verify thread_block.group_index() values
    ArrayAllOf(dim3_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.block_idx(i).value(); });

    HIP_CHECK(hipMemcpy(dim3_arr.ptr(), dim3_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*dim3_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_group_dims_getter<<<blocks, threads>>>(dim3_arr_dev.ptr());

    // Verify thread_block.thread_index() values
    ArrayAllOf(dim3_arr.ptr(), grid.thread_count_,
               [&grid](uint32_t i) { return grid.thread_idx(i).value(); });

    HIP_CHECK(hipMemcpy(dim3_arr.ptr(), dim3_arr_dev.ptr(),
                        grid.thread_count_ * sizeof(*dim3_arr.ptr()), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Verify thread_block.group_dim() values
    ArrayAllOf(dim3_arr.ptr(), grid.thread_count_, [threads](uint32_t) { return threads; });
  }
}

TEST_CASE("Unit_Thread_Block_Getters_Via_Base_Type") {
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(10, 10, 10));

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  thread_block_size_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  thread_block_thread_rank_getter<cg::thread_group><<<blocks, threads>>>(uint_arr_dev.ptr());

  // Verify thread_block.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.threads_in_block_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify thread_block.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
}

TEST_CASE("Unit_Thread_Block_Getters_Via_Non_Member_Functions") {
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(10, 10, 10));

  const CPUGrid grid(blocks, threads);

  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc,
                                              grid.thread_count_ * sizeof(unsigned int));
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc,
                                          grid.thread_count_ * sizeof(unsigned int));

  thread_block_non_member_size_getter<<<blocks, threads>>>(uint_arr_dev.ptr());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  thread_block_non_member_thread_rank_getter<<<blocks, threads>>>(uint_arr_dev.ptr());

  // Verify thread_block.size() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [size = grid.threads_in_block_count_](uint32_t) { return size; });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(),
                      grid.thread_count_ * sizeof(*uint_arr.ptr()), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify thread_block.thread_rank() values
  ArrayAllOf(uint_arr.ptr(), grid.thread_count_,
             [&grid](uint32_t i) { return grid.thread_rank_in_block(i).value(); });
}