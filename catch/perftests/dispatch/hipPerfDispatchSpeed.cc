/*
 Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* @addtogroup hipPerfDispatchSpeed hipPerfDispatchSpeed
* @{
* @ingroup perfDispatchTest
*/

#include <hip_test_common.hh>
#include <string.h>
#include <complex>

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif
#define CHAR_BUF_SIZE 512

typedef struct {
    unsigned int iterations;
    int flushEvery;
} testStruct;

testStruct testList[] = {
    { 1, -1},
    { 1, -1},
    { 10, 1},
    { 10, -1},
    { 100, 1},
    { 100, 10},
    { 100, -1},
    { 1000, 1},
    { 1000, 10},
    { 1000, 100},
    { 1000, -1},
    { 10000, 1},
    { 10000, 10},
    { 10000, 100},
    { 10000, 1000},
    { 10000, -1},
    { 100000, 1},
    { 100000, 10},
    { 100000, 100},
    { 100000, 1000},
    { 100000, 10000},
    { 100000, -1},
};

unsigned int mapTestList[] = {1, 1, 10, 100, 1000, 10000, 100000};

__global__ void _dispatchSpeed(float *outBuf) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < 0)
    outBuf[i] = 0.0f;
};

/**
* Test Description
* ------------------------
*  - Verify the hipPerf Dispatch speed.
* Test source
* ------------------------
*  - perftests/compute/hipPerfMandelbrot.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 5.6
*/

TEST_CASE("Perf_hipPerfDispatchSpeed") {
  int p_gpuDevice = 0;
  int p_tests = -1;
  hipError_t err = hipSuccess;
  hipDeviceProp_t props = {0};
  HIP_CHECK(hipGetDeviceProperties(&props, p_gpuDevice));

  unsigned int testListSize = sizeof(testList) / sizeof(testStruct);
  int numTests = (p_tests == -1) ? (2*2*testListSize - 1) : p_tests;
  int test = (p_tests == -1) ? 0 : p_tests;

  float* srcBuffer = NULL;
  unsigned int bufSize_ = 64*sizeof(float);
  err = hipMalloc(&srcBuffer, bufSize_);
  REQUIRE(err == hipSuccess);

  for (; test <= numTests; test++) {
    int openTest = test % testListSize;
    bool sleep = false;
    bool doWarmup = false;

    if ((test / testListSize) % 2) {
        doWarmup = true;
    }
    if (test >= (testListSize * 2)) {
        sleep = true;
    }
    int threads = (bufSize_ / sizeof(float));
    int threads_per_block  = 64;
    int blocks = (threads/threads_per_block) + (threads % threads_per_block);
    hipEvent_t start, stop;

    // NULL stream check:
    err = hipEventCreate(&start);
    REQUIRE(err == hipSuccess);
    err = hipEventCreate(&stop);
    REQUIRE(err == hipSuccess);

    if (doWarmup) {
      hipLaunchKernelGGL(_dispatchSpeed, dim3(blocks), dim3(threads_per_block),
                          0, hipStream_t(0), srcBuffer);
      err = hipDeviceSynchronize();
      REQUIRE(err == hipSuccess);
      }
      auto Start = std::chrono::high_resolution_clock::now();
      for (unsigned int i = 0; i < testList[openTest].iterations; i++) {
        HIP_CHECK(hipEventRecord(start, NULL));
        hipLaunchKernelGGL(_dispatchSpeed, dim3(blocks),
                        dim3(threads_per_block), 0, hipStream_t(0), srcBuffer);
        HIP_CHECK(hipEventRecord(stop, NULL));
        if ((testList[openTest].flushEvery > 0) &&
          (((i + 1) % testList[openTest].flushEvery) == 0)) {
            if (sleep) {
              err = hipDeviceSynchronize();
              REQUIRE(err == hipSuccess);
            } else {
              do {
                err = hipEventQuery(stop);
            } while (err == hipErrorNotReady);
          }
        }
      }
      if (sleep) {
        err = hipDeviceSynchronize();
        REQUIRE(err == hipSuccess);
      } else {
        do {
          err = hipEventQuery(stop);
        } while (err == hipErrorNotReady);
      }
      auto Stop = std::chrono::high_resolution_clock::now();
      HIP_CHECK(hipEventDestroy(start));
      HIP_CHECK(hipEventDestroy(stop));
      double sec = std::chrono::duration<double, std::milli>(Stop - Start).count();

      // microseconds per launch
      double perf = (1000000.f*sec/testList[openTest].iterations);
      const char *waitType;
      const char *extraChar;
      const char *n;
      const char *warmup;
      if (sleep) {
        waitType = "sleep";
        extraChar = "";
        n = "";
      } else {
        waitType = "spin";
        n = "n";
        extraChar = " ";
      }
      if (doWarmup) {
        warmup = "warmup";
      } else {
        warmup = "";
      }
      char buf[256];
      if (testList[openTest].flushEvery > 0) {
        SNPRINTF(buf, sizeof(buf), "HIPPerfDispatchSpeed[%3d] %7d dispatches %s%sing every %5d %6s (us/disp) %3f",
                 test, testList[openTest].iterations,
                 waitType, n, testList[openTest].flushEvery, warmup, (float)perf);
      } else {
        SNPRINTF(buf, sizeof(buf), "HIPPerfDispatchSpeed[%3d] %7d dispatches (%s%s)              %6s (us/disp) %3f",
                 test, testList[openTest].iterations,
                 waitType, extraChar, warmup, (float)perf);
      }
      printf("%s\n", buf);
  }
  HIP_CHECK(hipFree(srcBuffer));
}
