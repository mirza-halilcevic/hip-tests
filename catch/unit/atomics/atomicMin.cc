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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

template <typename T, bool shared = false>
__global__ void AtomicMin(T* const addr, const T val) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid == 0) ptr[0] = addr[0];
    __syncthreads();
  }

  atomicMin(ptr, val - tid);

  if constexpr (shared) {
    __syncthreads();
    if(tid == 0) addr[0] = ptr[0];
  }
}

template <typename T, bool shared = false>
__global__ void AtomicMinMultiDest(T* const addr, const T val, const int n) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid < n) ptr[tid] = addr[tid];
    __syncthreads();
  }

  atomicMin(ptr + tid % n , val - tid % n);

  if constexpr (shared) {
    __syncthreads();
    if (tid < n) addr[tid] = ptr[tid];
  }
}

TEMPLATE_TEST_CASE("Unit_atomicMin_Positive_SameAddress", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE(kValue - 2, kValue + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));
  HIP_CHECK(hipMemset(alloc.ptr(), kInitValue, 1));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicMin<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue + num_blocks * num_threads - 1);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicMin<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kValue + num_blocks * num_threads - 1);
  }

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = std::min(kInitValue, kValue);
  REQUIRE(res == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicMin_Positive_DifferentAddressSameWarp", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  int warp_size;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  const auto kSize = sizeof(TestType) * warp_size;
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE_REF(warp_size / 2 - 2, warp_size / 2 + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);
  TestType src[warp_size];
  for (int i = 0; i < warp_size; ++i) {
    src[i] = kInitValue;
  }
  HIP_CHECK(hipMemcpy(alloc.ptr(), src, kSize, hipMemcpyHostToDevice));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicMinMultiDest<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue + warp_size - 1, warp_size);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicMinMultiDest<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kValue + warp_size - 1, warp_size);
  }

  TestType res[warp_size];
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < warp_size; ++i) {
    const auto expected_res = std::min(kInitValue, kValue + warp_size - i - 1);
    REQUIRE(res[i] == expected_res);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicMin_Positive_MultiKernel", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE(kValue - 2, kValue + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));
  HIP_CHECK(hipMemset(alloc.ptr(), kInitValue, 1));

  StreamGuard stream1(Streams::created);
  StreamGuard stream2(Streams::created);

  int num_blocks = 3, num_threads = 128;
  HipTest::launchKernel(AtomicMin<TestType, false>, num_blocks, num_threads, 0, stream1.stream(),
                        alloc.ptr(), kValue + num_blocks * num_threads - 1);
  HipTest::launchKernel(AtomicMin<TestType, false>, num_blocks, num_threads, 0, stream2.stream(),
                        alloc.ptr(), kValue + num_blocks * num_threads - 1);

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = std::min(kInitValue, kValue);
  REQUIRE(res == expected_res);
}
