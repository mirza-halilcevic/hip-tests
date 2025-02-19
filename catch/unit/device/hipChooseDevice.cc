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

/**
 * @addtogroup hipChooseDevice hipChooseDevice
 * @{
 * @ingroup DeviceTest
 * `hipChooseDevice(int* device, const hipDeviceProp_t* prop)` -
 * Device which matches `hipDeviceProp_t` is returned.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate chosen device against gotten device properties.
 * Test source
 * ------------------------
 *  - unit/device/hipChooseDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipChooseDevice_ValidateDevId") {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = -1;
  HIP_CHECK(hipChooseDevice(&dev, &prop));
  REQUIRE_FALSE(dev < 0);
  REQUIRE_FALSE(dev >= numDevices);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the device is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When pointer to the properties is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipChooseDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipChooseDevice_NegTst") {
  hipDeviceProp_t prop;
  int dev = -1;

  SECTION("dev is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipChooseDevice(nullptr, &prop));
  }

  SECTION("prop is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipChooseDevice(&dev, nullptr));
  }
}
