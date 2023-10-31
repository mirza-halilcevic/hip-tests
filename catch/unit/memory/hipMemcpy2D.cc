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

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpy2D hipMemcpy2D
 * @{
 * @ingroup MemoryTest
 * `hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
 * size_t width, size_t height, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

/**
 * Test Description
 * ------------------------
 *  - Verifies basic test cases for copying 2D memory between
 *    device and host.
 *  - Validates following memcpy directions:
 *    -# Device to host
 *    -# Device to device
 *      - Peer access disabled
 *      - Peer access enabled
 *    -# Host to device
 *    -# Host to host
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2D_Positive_Basic") {
  constexpr bool async = false;

  SECTION("Device to Host") { Memcpy2DDeviceToHostShell<async>(hipMemcpy2D); }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") { Memcpy2DDeviceToDeviceShell<async, false>(hipMemcpy2D); }
    SECTION("Peer access enabled") { Memcpy2DDeviceToDeviceShell<async, true>(hipMemcpy2D); }
  }

  SECTION("Host to Device") { Memcpy2DHostToDeviceShell<async>(hipMemcpy2D); }

  SECTION("Host to Host") { Memcpy2DHostToHostShell<async>(hipMemcpy2D); }
}
/*
This testcase performs the following scenarios of hipMemcpy2D API on same GPU.
1. H2D-D2D-D2H for Host Memory<-->Device Memory
2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory
The src and dst input pointers to hipMemCpy2D add an offset to the pointers
returned by the allocation functions.

Input : "A_h" initialized based on data type
         "A_h" --> "A_d" using H2D copy
         "A_d" --> "B_d" using D2D copy
         "B_d" --> "B_h" using D2H copy
Output: Validating A_h with B_h both should be equal for
        the number of COLUMNS and ROWS copied
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_H2D-D2D-D2H_WithOffset", "", int, float, double) {
  CHECK_IMAGE_SUPPORT

  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType  *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
            *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

  // Allocating memory
  if (mem_type) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
  }
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d),
                          &pitch_B, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d+COLUMNS*sizeof(TestType), pitch_A, A_h, COLUMNS*sizeof(TestType),
                        COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d+COLUMNS*sizeof(TestType), pitch_B, A_d+COLUMNS*sizeof(TestType),
                        pitch_A, COLUMNS*sizeof(TestType),
                        ROWS, hipMemcpyDeviceToDevice));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS*sizeof(TestType), B_d+COLUMNS*sizeof(TestType), pitch_B,
                        COLUMNS*sizeof(TestType), ROWS,
                        hipMemcpyDeviceToHost));


  // Validating the result
  REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);


  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  if (mem_type) {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                  A_h, B_h, C_h, true);
  } else {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                  A_h, B_h, C_h, false);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that API synchronizes regarding to host when copying from
 *    device memory to the pageable or pinned host memory.
 *  - Validates following memcpy directions:
 *    -# Host to device
 *    -# Device to host
 *      - Pageable host memory
 *      - Pinned host memory
 *    -# Device to device
 *      - Platform specific (NVIDIA)
 *    -# Host to host
 *      - Platform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2D_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy2DHtoDSyncBehavior(hipMemcpy2D, true); }

  SECTION("Device to Host") {
    Memcpy2DDtoHPageableSyncBehavior(hipMemcpy2D, true);
    Memcpy2DDtoHPinnedSyncBehavior(hipMemcpy2D, true);
  }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-232
  SECTION("Device to Device") { Memcpy2DDtoDSyncBehavior(hipMemcpy2D, false); }

  SECTION("Host to Host") { Memcpy2DHtoHSyncBehavior(hipMemcpy2D, true); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates that nothing will be copied if width or height are set to zero.
 *  - Validates following memcpy directions:
 *    -# Device to host
 *    -# Device to device
 *    -# Host to device
 *    -# Host to host
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2D_Positive_Parameters") {
  constexpr bool async = false;
  Memcpy2DZeroWidthHeight<async>(hipMemcpy2D);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When destination pitch is less than width
 *      - Expected output: return `hipErrorInvalidPitchValue`
 *    -# When source pitch is less than width
 *      - Expected output: return `hipErrorInvalidPitchValue`
 *    -# When destination pitch is larger than maximum pitch
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pitch is larger than maximum pitch
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memcpy kind is not valid (-1)
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *  - All cases are executed for following memcpy directions:
 *    -# Host to device
 *    -# Device to host
 *    -# Host to host
 *    -# Device to device
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2D_Negative_Parameters") {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(nullptr, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, nullptr, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, width - 1, src, spitch, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, src, width - 1, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("dpitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, static_cast<size_t>(attr) + 1, src, spitch, width, height, kind),
          hipErrorInvalidValue);
    }

    SECTION("spitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, static_cast<size_t>(attr) + 1, width, height, kind),
          hipErrorInvalidValue);
    }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, spitch, width, height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  };

  SECTION("Host to Device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(host_alloc.ptr(), device_alloc.pitch(), device_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}
