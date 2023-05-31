/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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


/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <cstdlib>

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
#define ASSERT_EQUAL(lhs, rhs) HIPASSERT(lhs == rhs)
=======
#include "hip_cg_common.hh"
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc

using namespace cooperative_groups;

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
static __global__
void kernel_cg_thread_block_type(int *sizeTestD,
                                 int *thdRankTestD,
                                 int *syncTestD,
                                 dim3 *groupIndexTestD,
                                 dim3 *thdIndexTestD)
{
  thread_block tb = this_thread_block();
=======
static __global__ void kernel_cg_thread_block_type(int* size_dev, int* thd_rank_dev, int* sync_dev,
                                                   dim3* group_index_dev, dim3* thd_index_dev) {
  cg::thread_block tb = cg::this_thread_block();
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Test size
  sizeTestD[gIdx] = tb.size();

  // Test thread_rank
  thdRankTestD[gIdx] = tb.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tb.sync();
  syncTestD[gIdx] = sm[1] * sm[0];

  // Test group_index
  groupIndexTestD[gIdx] = tb.group_index();

  // Test thread_index
  thdIndexTestD[gIdx] = tb.thread_index();
}

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
static void test_cg_thread_block_type(int blockSize)
{
  int nBytes = sizeof(int) * 2 * blockSize;
  int nDim3Bytes = sizeof(dim3) * 2 * blockSize;
  int *sizeTestD, *sizeTestH;
  int *thdRankTestD, *thdRankTestH;
  int *syncTestD, *syncTestH;
  dim3 *groupIndexTestD, *groupIndexTestH;
  dim3 *thdIndexTestD, *thdIndexTestH;
=======
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

static void test_cg_thread_block_type(ThreadBlockTypeTests test_type, int block_size) {
  int num_bytes = sizeof(int) * 2 * block_size;
  int num_dim3_bytes = sizeof(dim3) * 2 * block_size;
  int *size_dev, *size_host;
  int *thd_rank_dev, *thd_rank_host;
  int *sync_dev, *sync_host;
  dim3 *group_index_dev, *group_index_host;
  dim3 *thd_index_dev, *thd_index_host;
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc

  // Allocate device memory
  HIPCHECK(hipMalloc(&sizeTestD, nBytes));
  HIPCHECK(hipMalloc(&thdRankTestD, nBytes));
  HIPCHECK(hipMalloc(&syncTestD, nBytes));
  HIPCHECK(hipMalloc(&groupIndexTestD, nDim3Bytes));
  HIPCHECK(hipMalloc(&thdIndexTestD, nDim3Bytes));

  // Allocate host memory
  HIPCHECK(hipHostMalloc(&sizeTestH, nBytes));
  HIPCHECK(hipHostMalloc(&thdRankTestH, nBytes));
  HIPCHECK(hipHostMalloc(&syncTestH, nBytes));
  HIPCHECK(hipHostMalloc(&groupIndexTestH, nDim3Bytes));
  HIPCHECK(hipHostMalloc(&thdIndexTestH, nDim3Bytes));

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_thread_block_type,
                     2,
                     blockSize,
                     0,
                     0,
                     sizeTestD,
                     thdRankTestD,
                     syncTestD,
                     groupIndexTestD,
                     thdIndexTestD);
=======
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
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc

  // Copy result from device to host
  HIPCHECK(hipMemcpy(sizeTestH, sizeTestD, nBytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(thdRankTestH, thdRankTestD, nBytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(groupIndexTestH, groupIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(thdIndexTestH, thdIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost));

  // Validate results for both blocks together
<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
  for (int i = 0; i < 2 * blockSize; ++i) {
    ASSERT_EQUAL(sizeTestH[i], blockSize);
    ASSERT_EQUAL(thdRankTestH[i], i % blockSize);
    ASSERT_EQUAL(syncTestH[i], 200);
    ASSERT_EQUAL(groupIndexTestH[i].x, (uint) i / blockSize);
    ASSERT_EQUAL(groupIndexTestH[i].y, 0);
    ASSERT_EQUAL(groupIndexTestH[i].z, 0);
    ASSERT_EQUAL(thdIndexTestH[i].x, (uint) i % blockSize);
    ASSERT_EQUAL(thdIndexTestH[i].y, 0);
    ASSERT_EQUAL(thdIndexTestH[i].z, 0);
=======
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
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc
  }

  // Free device memory
  HIPCHECK(hipFree(sizeTestD));
  HIPCHECK(hipFree(thdRankTestD));
  HIPCHECK(hipFree(syncTestD));
  HIPCHECK(hipFree(groupIndexTestD));
  HIPCHECK(hipFree(thdIndexTestD));

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
  //Free host memory
  HIPCHECK(hipHostFree(sizeTestH));
  HIPCHECK(hipHostFree(thdRankTestH));
  HIPCHECK(hipHostFree(syncTestH));
  HIPCHECK(hipHostFree(groupIndexTestH));
  HIPCHECK(hipHostFree(thdIndexTestH));
=======
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
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc
}

TEST_CASE("Unit_hipCGThreadBlockType") {
  // Use default device for validating the test
  int deviceId;
  hipDeviceProp_t deviceProperties;
  HIPCHECK(hipGetDevice(&deviceId));
  HIPCHECK(hipGetDeviceProperties(&deviceProperties, deviceId));

  if (!deviceProperties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

<<<<<<< HEAD:catch/unit/cooperativeGrps/hipCGThreadBlockType.cc
  // Test for blockSizes in powers of 2
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
  for (int blockSize = 2; blockSize <= maxThreadsPerBlock; blockSize = blockSize*2) {
    test_cg_thread_block_type(blockSize);
=======
  ThreadBlockTypeTests test_type = ThreadBlockTypeTests::basicApi;

  SECTION("Default thread block API test") { test_type = ThreadBlockTypeTests::basicApi; }

  SECTION("Base type thread block API test") { test_type = ThreadBlockTypeTests::baseType; }

  SECTION("Public API thread block test") { test_type = ThreadBlockTypeTests::publicApi; }

  // Test for blockSizes in powers of 2
  int max_threads_per_blk = device_properties.maxThreadsPerBlock;
  for (int block_size = 2; block_size <= max_threads_per_blk; block_size = block_size * 2) {
    test_cg_thread_block_type(test_type, block_size);
>>>>>>> 6d96fbc1f04b03e83091775177f7f40420a8c216:catch/unit/cooperativeGrps/hipCGThreadBlockType_old.cc
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_thread_block_type(max(2, rand() % maxThreadsPerBlock));
  }
}
