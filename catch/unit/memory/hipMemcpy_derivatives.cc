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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <memcpy1d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpyDtoH hipMemcpyDtoH
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes)` -
 * Copy data from Device to Host.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate array on host.
 *    -# Copy memory from Host to Device.
 *    -# Launch kernel.
 *    -# Copy memory from Device to Host.
 *    -# Validate results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoH_Positive_Basic") {
  MemcpyDeviceToHostShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  });
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the API synchronizes regarding to host when
 *    copying from device memory to pageable or pinned host memory.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoH_Positive_Synchronization_Behavior") {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  MemcpyDtoHPageableSyncBehavior(f, true);
  MemcpyDtoHPinnedSyncBehavior(f, true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoH_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}

/**
 * End doxygen group hipMemcpyDtoH.
 * @}
 */

/**
 * @addtogroup hipMemcpyHtoD hipMemcpyHtoD
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes)` -
 * Copy data from Host to Device.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate array on host.
 *    -# Copy memory from Host to Device.
 *    -# Launch kernel.
 *    -# Copy memory from Device to Host.
 *    -# Validate results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoD_Positive_Basic") {
  MemcpyHostToDeviceShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
  });
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the API synchronizes regarding to host when
 *    copying from pageable or pinned host memory to device memory.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoD_Positive_Synchronization_Behavior") {
  MemcpyHtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}

/**
 * End doxygen group hipMemcpyHtoD.
 * @}
 */

/**
 * @addtogroup hipMemcpyDtoD hipMemcpyDtoD
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)` -
 * Copy data from Device to Device.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate memory on the device and host.
 *    -# Allocate memory on another device.
 *    -# Launch kernel.
 *    -# Copy results from device to device.
 *    -# Copy results from device to host.
 *    -# Validate results.
 *  - Basic behavior is checked for following scenarios:
 *    -# Peer access enabled
 *    -# Peer access disabled
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoD_Positive_Basic") {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                         reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  SECTION("Peer access enabled") { MemcpyDeviceToDeviceShell<false, true>(f); }
  SECTION("Peer access disabled") { MemcpyDeviceToDeviceShell<false, false>(f); }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that API is asynchronous regarding to host when copying
 *    from device memory to device memory.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (NVIDIA)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoD_Positive_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
  return;
#endif
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      false);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}