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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <atomic>

/**
 * @addtogroup hipStreamAddCallback hipStreamAddCallback
 * @{
 * @ingroup StreamTest
 */

static constexpr size_t N = 4096;
static constexpr int numThreads = 1000;
static std::atomic<int> Cb_count{0}, Data_mismatch{0};
static hipStream_t mystream;
static float *A1_h, *C1_h;

#if HT_AMD
#define HIPRT_CB
#endif

static __global__ void device_function(float* C_d, float* A_d, size_t Num) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gputhread; i < Num; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }

  // Delay thread 1 only in the GPU
  if (gputhread == 1) {
    uint64_t wait_t = 3200000000, start = clock64(), cur;
    do {
      cur = clock64() - start;
    } while (cur < wait_t);
  }
}

static __global__ void device_function_gfx11(float* C_d, float* A_d, size_t Num) {
#if HT_AMD
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gputhread; i < Num; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }

  // Delay thread 1 only in the GPU
  if (gputhread == 1) {
    uint64_t wait_t = 3200000000, start = wall_clock64(), cur;
    do {
      cur = wall_clock64() - start;
    } while (cur < wait_t);
  }
#endif
}

static void HIPRT_CB Thread1_Callback(hipStream_t stream, hipError_t status,
                                      void* userData) {
  HIPASSERT(stream == mystream);
  HIPASSERT(userData == nullptr);
  HIPCHECK(status);

  for (size_t i = 0; i < N; i++) {
    // Validate the data and update Data_mismatch
    if (C1_h[i] != A1_h[i] * A1_h[i]) {
      Data_mismatch++;
    }
  }

  // Increment the Cb_count to indicate that the callback is processed.
  ++Cb_count;
}

static void HIPRT_CB Thread2_Callback(hipStream_t stream, hipError_t status,
                                      void* userData) {
  HIPASSERT(stream == mystream);
  HIPASSERT(userData == nullptr);
  HIPCHECK(status);

  for (size_t i = 0; i < N; i++) {
    // Validate the data and update Data_mismatch
    if (C1_h[i] != A1_h[i] * A1_h[i]) {
      Data_mismatch++;
    }
  }

  // Increment the Cb_count to indicate that the callback is processed.
  ++Cb_count;
}

void Thread1_func() {
  HIPCHECK(hipStreamAddCallback(mystream, Thread1_Callback, nullptr, 0));
}

void Thread2_func() {
  HIPCHECK(hipStreamAddCallback(mystream, Thread2_Callback, nullptr, 0));
}

/**
 * Test Description
 * ------------------------
 *  - Add callbacks on the streams from multiple threads.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamACb_MultiThread.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamAddCallback_MultipleThreads") {
  float *A_d, *C_d;
  size_t Nbytes = (N) * sizeof(float);
  constexpr float Phi = 1.618f;

  A1_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A1_h != nullptr);
  C1_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(C1_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A1_h[i] = Phi + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  HIP_CHECK(
  hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

  HIP_CHECK(
  hipMemcpyAsync(A_d, A1_h, Nbytes, hipMemcpyHostToDevice,
                 mystream));

  constexpr unsigned threadsPerBlock = 256;
  constexpr unsigned blocks = (N + 255)/threadsPerBlock;

  auto device_function_used = IsGfx11() ? device_function_gfx11 : device_function;
  hipLaunchKernelGGL((device_function_used), dim3(blocks),
                     dim3(threadsPerBlock), 0,
                     mystream, C_d, A_d, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(
  hipMemcpyAsync(C1_h, C_d, Nbytes,
                 hipMemcpyDeviceToHost, mystream));

  std::thread *T = new std::thread[numThreads];
  for (int i = 0; i < numThreads; i++) {
    // Use different callback for every even thread
    // The callbacks will be added to same stream from different threads
    if ((i%2) == 0)
      T[i] = std::thread(Thread1_func);
    else
      T[i] = std::thread(Thread2_func);
  }

  // Wait until all the threads finish their execution
  for (int i = 0; i < numThreads; i++) {
    T[i].join();
  }

  HIP_CHECK(hipStreamSynchronize(mystream));
  HIP_CHECK(hipStreamDestroy(mystream));

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));

  free(A1_h);
  free(C1_h);

  // Cb_count should match total number of callbacks added from both threads
  // Data_mismatch will be updated if there is problem in data validation
  REQUIRE(Cb_count.load() == numThreads);
  REQUIRE(Data_mismatch.load() == 0);
  delete[] T;
}
