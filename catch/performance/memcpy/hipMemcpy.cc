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

#include "memcpy_performance_common.hh"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for all memcpy HIP APIs.
 */

class MemcpyBenchmark : public Benchmark<MemcpyBenchmark> {
 public:
  void operator()(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemcpy(dst, src, size, kind));
    }
  }
};

static void RunBenchmark(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type,
                         size_t size, hipMemcpyKind kind, bool enable_peer_access=false) {
  MemcpyBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(src_allocation_type));
  benchmark.AddSectionName(GetAllocationSectionName(dst_allocation_type));

  if (kind != hipMemcpyDeviceToDevice) {
    LinearAllocGuard<int> src_allocation(src_allocation_type, size);
    LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);
    benchmark.Run(dst_allocation.ptr(), src_allocation.ptr(), size, kind);
  } else {
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));

    LinearAllocGuard<int> src_allocation(src_allocation_type, size);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);
    HIP_CHECK(hipSetDevice(src_device));
    benchmark.Run(dst_allocation.ptr(), src_allocation.ptr(), size, kind);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy_DeviceToHost") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy_HostToDevice") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy_HostToHost") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy_DeviceToDevice_EnablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice, true);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy_DeviceToDevice_DisablePeerAccess") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice);
}
