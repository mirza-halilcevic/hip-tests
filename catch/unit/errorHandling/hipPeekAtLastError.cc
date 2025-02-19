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
#include <threaded_zig_zag_test.hh>

/**
 * @addtogroup hipPeekAtLastError hipPeekAtLastError
 * @{
 * @ingroup ErrorTest
 * `hipPeekAtLastError(void)` -
 * Return last error returned by any HIP runtime API call.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that `hipErrorInvalidValue` is returned after invalid `hipMalloc` call.
 *  - Validate that `hipSuccess` is returned again for getting the last error.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPeekAtLastError_Positive_Basic") {
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK_ERROR(hipMalloc(nullptr, 1), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
  HIP_CHECK(hipPeekAtLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Validate that appropriate error is returned when working with multiple threads.
 *  - Validate that appropriate error is returned for getting the last erro when working with multiple threads.
 *  - Cause error on purpose within one of the threads.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPeekAtLastError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPeekAtLastError_Positive_Threaded") {
  class HipPeekAtLastErrorTest : public ThreadedZigZagTest<HipPeekAtLastErrorTest> {
   public:
    void TestPart2() { REQUIRE_THREAD(hipMalloc(nullptr, 1) == hipErrorInvalidValue); }
    void TestPart3() {
      HIP_CHECK(hipPeekAtLastError());
      HIP_CHECK(hipGetLastError());
    }
    void TestPart4() { REQUIRE_THREAD(hipPeekAtLastError() == hipErrorInvalidValue); }
  };

  HipPeekAtLastErrorTest test;
  test.run();
}
