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

namespace cg = cooperative_groups;

enum class ThreadBlockTypeTests { basicApi, baseType, publicApi };

__device__ unsigned int thread_rank_in_grid() {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  return block_rank_in_grid * block_size + thread_rank_in_block;
}

static __global__ void thread_block_size_getter(unsigned int* sizes) {
  sizes[thread_rank_in_grid()] = cg::this_thread_block().size();
}

static __global__ void thread_block_thread_rank_getter(unsigned int* thread_ranks) {
  thread_ranks[thread_rank_in_grid()] = cg::this_thread_block().thread_rank();
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

static __global__ void sync_temp(int* sync_dev) {
  // Test sync
  // __shared__ int sm[2];
  // if (threadIdx.x == 0)
  //   sm[0] = 10;
  // else if (threadIdx.x == 1)
  //   sm[1] = 20;
  // tb.sync();
  // sync_dev[gIdx] = sm[1] * sm[0];
}

static __global__ void kernel_cg_thread_block_type_via_base_type(int* size_dev, int* thd_rank_dev,
                                                                 int* sync_dev) {
  cg::thread_group tg = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  size_dev[gIdx] = tg.size();

  // Test thread_rank
  thd_rank_dev[gIdx] = tg.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tg.sync();
  sync_dev[gIdx] = sm[1] * sm[0];
}

static __global__ void kernel_cg_thread_block_type_via_public_api(int* size_dev, int* thd_rank_dev,
                                                                  int* sync_dev) {
  cg::thread_block tb = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  size_dev[gIdx] = cg::group_size(tb);

  // Test thread_rank api
  thd_rank_dev[gIdx] = cg::thread_rank(tb);

  // Test sync api
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  cg::sync(tb);
  sync_dev[gIdx] = sm[1] * sm[0];
}

static bool operator==(const dim3& l, const dim3& r) {
  return l.x == r.x && l.y == r.y && l.z == r.z;
}

static bool operator!=(const dim3& l, const dim3& r) { return !(l == r); }

struct CPUGrid {
  CPUGrid(const dim3 grid_dim, const dim3 block_dim)
      : grid_dim_{grid_dim},
        block_dim_{block_dim},
        block_count_{grid_dim.x * grid_dim.y * grid_dim.z},
        threads_in_block_count_{block_dim.x * block_dim.y * block_dim.z},
        thread_count_{block_count_ * threads_in_block_count_} {}

  std::optional<unsigned int> thread_rank_in_block(const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    return thread_rank_in_grid -
        (thread_rank_in_grid / threads_in_block_count_) * threads_in_block_count_;
  }

  std::optional<dim3> block_idx(const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    dim3 block_idx;
    const auto block_rank_in_grid = thread_rank_in_grid / threads_in_block_count_;
    block_idx.x = block_rank_in_grid % grid_dim_.x;
    block_idx.y = (block_rank_in_grid / grid_dim_.x) % grid_dim_.y;
    block_idx.z = block_rank_in_grid / (grid_dim_.x * grid_dim_.y);

    return block_idx;
  }

  std::optional<dim3> thread_idx(const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    dim3 thread_idx;
    const auto thread_rank_in_block = thread_rank_in_grid % threads_in_block_count_;
    thread_idx.x = thread_rank_in_block % block_dim_.x;
    thread_idx.y = (thread_rank_in_block / block_dim_.x) % block_dim_.y;
    thread_idx.z = thread_rank_in_block / (block_dim_.x * block_dim_.y);

    return thread_idx;
  }

  const dim3 grid_dim_;
  const dim3 block_dim_;
  const unsigned int block_count_;
  const unsigned int threads_in_block_count_;
  const unsigned int thread_count_;
};

template <typename T, typename F>
static inline void ArrayAllOf(const T* arr, uint32_t count, F value_gen) {
  for (auto i = 0u; i < count; ++i) {
    const auto expected_val = value_gen(i);
    // Using require on every iteration leads to a noticeable performance loss on large arrays, even
    // when the require passes.
    if (arr[i] != expected_val) {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] == expected_val);
    }
  }
}

TEST_CASE("Unit_Thread_Block_Getters_Positive_Basic") {
  auto threads = GENERATE(dim3(256, 2, 2));
  auto blocks = GENERATE(dim3(10, 10, 10));

  const CPUGrid grid(blocks, threads);

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

/*
static void test_cg_thread_block_type(ThreadBlockTypeTests test_type, int block_size) {
  int num_bytes = sizeof(int) * 2 * block_size;
  int num_dim3_bytes = sizeof(dim3) * 2 * block_size;
  int *size_dev, *size_host;
  int *thd_rank_dev, *thd_rank_host;
  int *sync_dev, *sync_host;
  dim3 *group_index_dev, *group_index_host;
  dim3 *thd_index_dev, *thd_index_host;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&size_dev, num_bytes));
  HIP_CHECK(hipMalloc(&thd_rank_dev, num_bytes));
  HIP_CHECK(hipMalloc(&sync_dev, num_bytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc(&size_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&thd_rank_host, num_bytes));
  HIP_CHECK(hipHostMalloc(&sync_host, num_bytes));

  switch (test_type) {
    case (ThreadBlockTypeTests::basicApi):
      HIP_CHECK(hipMalloc(&group_index_dev, num_dim3_bytes));
      HIP_CHECK(hipMalloc(&thd_index_dev, num_dim3_bytes));
      HIP_CHECK(hipHostMalloc(&group_index_host, num_dim3_bytes));
      HIP_CHECK(hipHostMalloc(&thd_index_host, num_dim3_bytes));

      hipLaunchKernelGGL(kernel_cg_thread_block_type, 2, block_size, 0, 0, size_dev, thd_rank_dev,
                         sync_dev, group_index_dev, thd_index_dev);
      break;
    case (ThreadBlockTypeTests::baseType):
      hipLaunchKernelGGL(kernel_cg_thread_block_type_via_base_type, 2, block_size, 0, 0, size_dev,
                         thd_rank_dev, sync_dev);
      break;
    case (ThreadBlockTypeTests::publicApi):
      hipLaunchKernelGGL(kernel_cg_thread_block_type_via_public_api, 2, block_size, 0, 0, size_dev,
                         thd_rank_dev, sync_dev);
  }

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(size_host, size_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(thd_rank_host, thd_rank_dev, num_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sync_host, sync_dev, num_bytes, hipMemcpyDeviceToHost));
  if (test_type == ThreadBlockTypeTests::basicApi) {
    HIP_CHECK(hipMemcpy(group_index_host, group_index_dev, num_dim3_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(thd_index_host, thd_index_dev, num_dim3_bytes, hipMemcpyDeviceToHost));
  }

  // Validate results for both blocks together
  for (int i = 0; i < 2 * block_size; ++i) {
    ASSERT_EQUAL(size_host[i], block_size);
    ASSERT_EQUAL(thd_rank_host[i], i % block_size);
    ASSERT_EQUAL(sync_host[i], 200);
    if (test_type == ThreadBlockTypeTests::basicApi) {
      ASSERT_EQUAL(group_index_host[i].x, (uint)i / block_size);
      ASSERT_EQUAL(group_index_host[i].y, 0);
      ASSERT_EQUAL(group_index_host[i].z, 0);
      ASSERT_EQUAL(thd_index_host[i].x, (uint)i % block_size);
      ASSERT_EQUAL(thd_index_host[i].y, 0);
      ASSERT_EQUAL(thd_index_host[i].z, 0);
    }
  }

  // Free device memory
  HIP_CHECK(hipFree(size_dev));
  HIP_CHECK(hipFree(thd_rank_dev));
  HIP_CHECK(hipFree(sync_dev));

  // Free host memory
  HIP_CHECK(hipHostFree(size_host));
  HIP_CHECK(hipHostFree(thd_rank_host));
  HIP_CHECK(hipHostFree(sync_host));

  if (test_type == ThreadBlockTypeTests::basicApi) {
    HIP_CHECK(hipFree(group_index_dev));
    HIP_CHECK(hipFree(thd_index_dev));
    HIP_CHECK(hipHostFree(group_index_host));
    HIP_CHECK(hipHostFree(thd_index_host));
  }
}


TEST_CASE("Unit_hipCGThreadBlockType") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  ThreadBlockTypeTests test_type = ThreadBlockTypeTests::basicApi;

  SECTION("Default thread block API test") { test_type = ThreadBlockTypeTests::basicApi; }

  SECTION("Base type thread block API test") { test_type = ThreadBlockTypeTests::baseType; }

  SECTION("Public API thread block test") { test_type = ThreadBlockTypeTests::publicApi; }

  // Test for blockSizes in powers of 2
  int max_threads_per_blk = device_properties.maxThreadsPerBlock;
  for (int block_size = 2; block_size <= max_threads_per_blk; block_size = block_size * 2) {
    test_cg_thread_block_type(test_type, block_size);
  }

  // Test for random block_size, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_thread_block_type(test_type, max(2, rand() % max_threads_per_blk));
  }
}
*/

TEST_CASE("Blahem") {
  CPUGrid cpu_grid(dim3(2, 2, 2), dim3(2, 2, 2));
  for (auto i = 0; i < cpu_grid.thread_count_; ++i) {
    const auto block_idx = cpu_grid.block_idx(i).value();
    const auto thread_idx = cpu_grid.thread_idx(i).value();
    std::cout << "(" << block_idx.x << ", " << block_idx.y << ", " << block_idx.z << ")"
              << " ";
    std::cout << "(" << thread_idx.x << ", " << thread_idx.y << ", " << thread_idx.z << ")"
              << std::endl;
  }
}