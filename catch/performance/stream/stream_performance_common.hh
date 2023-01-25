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

#pragma once

#include <hip_test_common.hh>
#include <performance_common.hh>

static int IsStreamWaitValueSupported(int device_id) {
  int wait_value_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&wait_value_supported,
                                  hipDeviceAttributeCanUseStreamWaitValue, 0));
  return wait_value_supported;
}

static std::string GetFlagWaitSectionName(unsigned int flag) {
  if (flag == hipStreamWaitValueGte) {
    return "greater than";
  } else if (flag == hipStreamWaitValueEq) {
    return "equal";
  } else if (flag == hipStreamWaitValueAnd) {
    return "logical and";
  } else if (flag == hipStreamWaitValueNor) {
    return "logical nor";
  } else {
    return "unknown flag";
  }
}

static int AreMemPoolsSupported(int device_id) {
  int mem_pools_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pools_supported,
                                  hipDeviceAttributeMemoryPoolsSupported, 0));
  return mem_pools_supported;
}

static constexpr hipMemPoolProps kPoolProps = {
  hipMemAllocationTypePinned,
  hipMemHandleTypeNone,
  {
    hipMemLocationTypeDevice,
    0
  },
  nullptr,
  {0}
};
