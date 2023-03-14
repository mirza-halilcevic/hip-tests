/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include "launch_bounds_negative_kernels_rtc.hh"

#define MAX_THREADS_PER_BLOCK 128
#define MIN_WARPS_PER_MULTIPROCESSOR 2147483647

__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_MULTIPROCESSOR)
__global__ void SumKernel(int* sum) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(sum, tid);
}

template<bool out_of_bounds> void LaunchBoundsWrapper(const int threads_per_block) {
  auto block_size = GENERATE(1, 32, 128);
  int* A_d;
  int* A_h;
  int sum{0};

  A_h = static_cast<int*>(malloc(sizeof(int)));
  memset(A_h, 0, sizeof(int));
  HIP_CHECK(hipMalloc(&A_d, sizeof(int)));
  HIP_CHECK(hipMemcpy(A_d, A_h, sizeof(int), hipMemcpyHostToDevice));
  SumKernel<<<block_size, threads_per_block>>>(A_d);

  if constexpr (out_of_bounds) {
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorLaunchFailure);
  } else {
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipMemcpy(A_h, A_d, sizeof(int), hipMemcpyDeviceToHost));

  if constexpr (!out_of_bounds) {
    for (int i = 0; i < threads_per_block * block_size; ++i) {
      sum += i;
    }
    REQUIRE(*A_h == sum);
  }

  free(A_h);
  HIP_CHECK(hipFree(A_d));
}

TEST_CASE("Unit_Kernel_Launch_bounds_Positive_Basic") {
  auto threads_per_block = GENERATE(1, MAX_THREADS_PER_BLOCK / 2, MAX_THREADS_PER_BLOCK);
  LaunchBoundsWrapper<false>(threads_per_block);
}

TEST_CASE("Unit_Kernel_Launch_bounds_Negative_OutOfBounds") {
  auto threads_per_block = GENERATE(MAX_THREADS_PER_BLOCK + 1, 2 * MAX_THREADS_PER_BLOCK);
  LaunchBoundsWrapper<true>(threads_per_block);
}

TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source = kMinWarpsNotInt;
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "launch_bounds_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log.
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  std::cout << log << std::endl;

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
}
