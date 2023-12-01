/*
Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipMemcpy2DAsync hipMemcpy2DAsync
 * @{
 * @ingroup MemcpyTest
 * `hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
 *                   size_t spitch, size_t width, size_t height,
 *                   hipMemcpyKind kind, hipStream_t stream = 0 )` -
 * Copies data between host and device.
 */

// Testcase Description:
// 1) Verifies the working of Memcpy2DAsync API negative scenarios by
//    Pass NULL to destination pointer
//    Pass NULL to Source pointer
//    Pass width greater than spitch/dpitch
// 2) Verifies hipMemcpy2DAsync API by
//    pass 0 to destionation pitch
//    pass 0 to source pitch
//    pass 0 to width
//    pass 0 to height
// 3) Verifies working of Memcpy2DAsync API on host memory
//    and pinned host memory by
//    performing D2H, D2D and H2D memory kind copies on same GPU
// 4) Verifies working of Memcpy2DAsync API on host memory
//    and pinned host memory by
//    performing D2H, D2D and H2D memory kind copies on peer GPU
// 5) Verifies working of Memcpy2DAsync API where memory is allocated
//    in GPU-0 and stream is created on GPU-1

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr auto NUM_W{16};
static constexpr auto NUM_H{16};
static constexpr auto COLUMNS{6};
static constexpr auto ROWS{6};

/**
 * Test Description
 * ------------------------
 *  - This performs the following scenarios of hipMemcpy2DAsync API on same GPU
      1. H2D-D2D-D2H for Host Memory<-->Device Memory
      2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory

      Input : "A_h" initialized based on data type
         "A_h" --> "A_d" using H2D copy
         "A_d" --> "B_d" using D2D copy
         "B_d" --> "B_h" using D2H copy
      Output: Validating A_h with B_h both should be equal for
        the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2DAsync_Host&PinnedMem", ""
                   , int, float, double) {
  CHECK_IMAGE_SUPPORT
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType  *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
            *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating memory
  if (mem_type) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
  }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy2DDeviceToDeviceShell<async, false>(
          std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, stream), stream);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDeviceToDeviceShell<async, true>(
          std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, stream), stream);
    }
  }
  SECTION("Calling Async apis with hipStreamPerThread") {
    // Host to Device
    HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
                               COLUMNS*sizeof(TestType), ROWS,
                               hipMemcpyHostToDevice, hipStreamPerThread));

    // Performs D2D on same GPU device
    HIP_CHECK(hipMemcpy2DAsync(B_d, pitch_B, A_d, pitch_A,
                               COLUMNS*sizeof(TestType), ROWS,
                               hipMemcpyDeviceToDevice, hipStreamPerThread));

  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<async>(
        std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, stream), stream);
  }

  SECTION("Host to Host") {
    Memcpy2DHostToHostShell<async>(std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, stream),
                                   stream);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcases performs the following scenarios of hipMemcpy2DAsync API on Peer GPU
      1. H2D-D2D-D2H for Host Memory<-->Device Memory
      2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory

      Input : "A_h" initialized based on data type
               "A_h" --> "A_d" using H2D copy
               "A_d" --> "X_d" using D2D copy
               "X_d" --> "B_h" using D2H copy
      Output: Validating A_h with B_h both should be equal for
              the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2DAsync_multiDevice-Host&PinnedMem", ""
                   , int, float, double) {
  CHECK_IMAGE_SUPPORT
  auto mem_type = GENERATE(0, 1);
  int numDevices = 0;
  int canAccessPeer = 0;
  TestType* A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  hipStream_t stream;

  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipStreamCreate(&stream));

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

      // Initialize the data
      HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

      // Host to Device
      HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
            COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice, stream));

      // Change device
      HIP_CHECK(hipSetDevice(1));

      char *X_d{nullptr};
      size_t pitch_X;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d),
            &pitch_X, width, NUM_H));

      // Device to Device
      HIP_CHECK(hipMemcpy2DAsync(X_d, pitch_X, A_d,
            pitch_A, COLUMNS*sizeof(TestType),
            ROWS, hipMemcpyDeviceToDevice, stream));

      // Device to Host
      HIP_CHECK(hipMemcpy2DAsync(B_h, COLUMNS*sizeof(TestType), X_d,
                                 pitch_X, COLUMNS*sizeof(TestType), ROWS,
                                 hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      if (mem_type) {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, true);
      } else {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, false);
      }
      HIP_CHECK(hipFree(X_d));
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcases performs the following scenarios of hipMemcpy2DAsync API on Peer GPU
      1. H2D-D2D-D2H for Host Memory<-->Device Memory
      2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory
      Memory is allocated in GPU-0 and Stream is created in GPU-1

      Input : "A_h" initialized based on data type
               "A_h" --> "A_d" using H2D copy
               "A_d" --> "X_d" using D2D copy
               "X_d" --> "B_h" using D2H copy
      Output: Validating A_h with B_h both should be equal for
              the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2DAsync_multiDevice-StreamOnDiffDevice", ""
                   , int, float, double) {
  CHECK_IMAGE_SUPPORT
  auto mem_type = GENERATE(0, 1);
  int numDevices = 0;
  int canAccessPeer = 0;
  TestType* A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  hipStream_t stream;

  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));

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
      char *X_d{nullptr};
      size_t pitch_X;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d),
            &pitch_X, width, NUM_H));

      // Initialize the data
      HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

      // Change device
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&stream));

      // Host to Device
      HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
            COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice, stream));

      // Device to Device
      HIP_CHECK(hipMemcpy2DAsync(X_d, pitch_X, A_d,
            pitch_A, COLUMNS*sizeof(TestType),
            ROWS, hipMemcpyDeviceToDevice, stream));

      // Device to Host
      HIP_CHECK(hipMemcpy2DAsync(B_h, COLUMNS*sizeof(TestType), X_d,
                                 pitch_X, COLUMNS*sizeof(TestType), ROWS,
                                 hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      if (mem_type) {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, true);
      } else {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, false);
      }
      HIP_CHECK(hipFree(X_d));
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the null checks of hipMemcpy2DAsync API
      1. hipMemcpy2DAsync API where Source Pitch is zero
      2. hipMemcpy2DAsync API where Destination Pitch is zero
      3. hipMemcpy2DAsync API where height is zero
      4. hipMemcpy2DAsync API where width is zero
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEST_CASE("Unit_hipMemcpy2DAsync_SizeCheck") {
  CHECK_IMAGE_SUPPORT
  HIP_CHECK(hipSetDevice(0));
  int* A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-233
  SECTION("Device to Pageable Host") {
    Memcpy2DDtoHPageableSyncBehavior(
        std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, nullptr), true);
  }
#endif

  SECTION("Device to Pinned Host") {
    Memcpy2DDtoHPinnedSyncBehavior(std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, nullptr),
                                   false);
  }

  SECTION("Device to Device") {
    Memcpy2DDtoDSyncBehavior(std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, nullptr),
                             false);
  }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-233
  SECTION("Host to Host") {
    Memcpy2DHtoHSyncBehavior(std::bind(hipMemcpy2DAsync, _1, _2, _3, _4, _5, _6, _7, nullptr),
                             true);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs the negative scenarios of hipMemcpy2DAsync API
      1. hipMemcpy2DAsync API by Passing nullptr to destination
      2. hipMemcpy2DAsync API by Passing nullptr to source
      3. hipMemcpy2DAsync API where width is > destination pitch
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */

TEST_CASE("Unit_hipMemcpy2DAsync_Negative") {
  CHECK_IMAGE_SUPPORT
  HIP_CHECK(hipSetDevice(0));
  int* A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating memory
  HipTest::initArrays<int>(nullptr, nullptr, nullptr,
      &A_h, nullptr, nullptr, NUM_W*NUM_H);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
        &pitch_A, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<int>(NUM_W*NUM_H, A_h, nullptr, nullptr);

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
 *    -# When stream is not valid
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorContextIsDestroyed`
 *  - All cases are executed for following memcpy directions:
 *    -# Host to device
 *    -# Device to host
 *    -# Host to host
 *    -# Device to device
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2DAsync_Negative_Parameters") {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DAsync(nullptr, dpitch, src, spitch, width, height, kind, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, dpitch, nullptr, spitch, width, height, kind, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, width - 1, src, spitch, width, height, kind, nullptr),
                      hipErrorInvalidPitchValue);
    }
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, dpitch, src, width - 1, width, height, kind, nullptr),
                      hipErrorInvalidPitchValue);
    }
    SECTION("dpitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, static_cast<size_t>(attr) + 1, src, spitch, width,
                                       height, kind, nullptr),
                      hipErrorInvalidValue);
    }
    SECTION("spitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, dpitch, src, static_cast<size_t>(attr) + 1, width,
                                       height, kind, nullptr),
                      hipErrorInvalidValue);
    }
#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                       static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }
#endif
#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-235
    SECTION("Invalid stream") {
      StreamGuard stream_guard(Streams::created);
      HIP_CHECK(hipStreamDestroy(stream_guard.stream()));
      HIP_CHECK_ERROR(
          hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream_guard.stream()),
          hipErrorContextIsDestroyed);
    }
#endif
  };

  SECTION("Host to device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("hipMemcpy2DAsync API by Passing nullptr to source") {
    REQUIRE(hipMemcpy2DAsync(A_h, width, nullptr,
            pitch_A, COLUMNS*sizeof(int), ROWS,
            hipMemcpyDeviceToHost, stream) != hipSuccess);
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}

static void hipMemcpy2DAsync_Basic_Size_Test(size_t inc) {
  constexpr int defaultProgramSize = 256 * 1024 * 1024;
  constexpr int N = 2;
  constexpr int value = 42;
  int *in, *out, *dev;
  size_t newSize = 0, inp = 0;
  size_t size = sizeof(int) * N * inc;

  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  if ( free < 2 * size )
    newSize = ( free - defaultProgramSize ) / 2;
  else
    newSize = size;

  INFO("Array size: " << size/1024.0/1024.0 << " MB or " << size << " Bytes.");
  INFO("Free memory: " << free/1024.0/1024.0 << " MB or " << free << " Bytes");
  INFO("NewSize:" << newSize/1024.0/1024.0 << "MB or " << newSize << " Bytes");

  HIP_CHECK(hipHostMalloc(&in, newSize));
  HIP_CHECK(hipHostMalloc(&out, newSize));
  HIP_CHECK(hipMalloc(&dev, newSize));

  inp = newSize / (sizeof(int) * N);
  for (size_t i=0; i < N; i++) {
    in[i * inp] = value;
  }

  size_t pitch = sizeof(int) * inp;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpy2DAsync(dev, pitch, in, pitch, sizeof(int),
                             N, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpy2DAsync(out, pitch, dev, pitch, sizeof(int),
                             N, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i=0; i < N; i++) {
    REQUIRE(out[i * inp] == value);
  }

  HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipHostFree(in));
  HIP_CHECK(hipHostFree(out));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs multidevice size check on hipMemcpy2DAsync API
      1. Verify hipMemcpy2DAsync with 1 << 20 size
      2. Verify hipMemcpy2DAsync with 1 << 21 size
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipMemcpy2DAsync_multiDevice_Basic_Size_Test") {
  CHECK_IMAGE_SUPPORT
  size_t input = 1 << 20;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  for (int i=0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));

    SECTION("Verify hipMemcpy2DAsync with 1 << 20 size") {
      hipMemcpy2DAsync_Basic_Size_Test(input);
    }
    SECTION("Verify hipMemcpy2DAsync with 1 << 21 size") {
      input <<= 1;
      hipMemcpy2DAsync_Basic_Size_Test(input);
    }
  }
}
