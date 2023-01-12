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
#include <performance_common.hh>
#include <resource_guards.hh>

static hip_Memcpy2D CreateMemcpy2DParam(void* dst, size_t dpitch, void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream=nullptr) {
  hip_Memcpy2D params = {0};
  const hipExtent src_offset = {0};
  const hipExtent dst_offset = {0};
  params.dstPitch = dpitch;
  switch (kind) {
    case hipMemcpyDeviceToHost:
    case hipMemcpyHostToHost:
      #if HT_AMD
        params.dstMemoryType = hipMemoryTypeHost;
      #else
        params.dstMemoryType = CU_MEMORYTYPE_HOST;
      #endif
      params.dstHost = dst;
      break;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyHostToDevice:
      #if HT_AMD
        params.dstMemoryType = hipMemoryTypeDevice;
      #else
        params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      #endif
      params.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
      break;
    default:
      REQUIRE(false);
  }

  params.srcPitch = dpitch;
  switch (kind) {
    case hipMemcpyDeviceToHost:
    case hipMemcpyHostToHost:
      #if HT_AMD
        params.srcMemoryType = hipMemoryTypeHost;
      #else
        params.srcMemoryType = CU_MEMORYTYPE_HOST;
      #endif
      params.srcHost = src;
      break;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyHostToDevice:
      #if HT_AMD
        params.srcMemoryType = hipMemoryTypeDevice;
      #else
        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      #endif
      params.srcDevice = reinterpret_cast<hipDeviceptr_t>(src);
      break;
    default:
      REQUIRE(false);
  }

  params.WidthInBytes = width;
  params.Height = height;
  params.srcXInBytes = src_offset.width;
  params.srcY = src_offset.height;
  params.dstXInBytes = dst_offset.width;
  params.dstY = dst_offset.height;

  return params;
}

class MemcpyParam2DBenchmark : public Benchmark<MemcpyParam2DBenchmark> {
 public:
  void operator()(size_t width, size_t height, hipMemcpyKind kind, bool enable_peer_access) {
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();
  
    if (kind == hipMemcpyDeviceToHost) {
      LinearAllocGuard2D<int> device_allocation(width, height);
      const size_t host_pitch = GENERATE_REF(device_allocation.width(),
                    device_allocation.width() + device_allocation.height() / 2);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, host_pitch * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(host_allocation.ptr(), host_pitch,
                           device_allocation.ptr(), device_allocation.pitch(),
                           device_allocation.width(), device_allocation.height(),
                           kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpyParam2DAsync(&params, stream));
      }
    } else if (kind == hipMemcpyHostToDevice) {
      LinearAllocGuard2D<int> device_allocation(width, height);
      const size_t host_pitch = GENERATE_REF(device_allocation.width(),
                    device_allocation.width() + device_allocation.height() / 2);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, host_pitch * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(device_allocation.ptr(), device_allocation.pitch(),
                           host_allocation.ptr(), host_pitch,
                           device_allocation.width(), device_allocation.height(),
                           kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpyParam2DAsync(&params, stream));
      }
    } else if (kind == hipMemcpyHostToHost) {
      const size_t src_pitch = GENERATE_REF(width * sizeof(int), width * sizeof(int) + height / 2); 
      LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc, src_pitch * height);
      LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(dst_allocation.ptr(), width * sizeof(int),
                           src_allocation.ptr(), src_pitch, width * sizeof(int), height,
                           kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpyParam2DAsync(&params, stream));
      }
    } else {
      // hipMemcpyDeviceToDevice
      int src_device = 0;
      int dst_device = 1;

      if (enable_peer_access) {
        int can_access_peer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
        if (!can_access_peer) {
          INFO("Peer access cannot be enabled between devices " << src_device << " and " << dst_device);
          REQUIRE(can_access_peer);
        }
        HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));
      }
      LinearAllocGuard2D<int> src_allocation(width, height);
      HIP_CHECK(hipSetDevice(dst_device));
      LinearAllocGuard2D<int> dst_allocation(width, height);

      HIP_CHECK(hipSetDevice(src_device));
      hip_Memcpy2D params = CreateMemcpy2DParam(dst_allocation.ptr(), dst_allocation.pitch(),
                           src_allocation.ptr(), src_allocation.pitch(),
                           dst_allocation.width(), dst_allocation.height(),
                           kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpyParam2DAsync(&params, stream));
      }
    }

    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind, bool enable_peer_access=false) {
  MemcpyParam2DBenchmark benchmark;
  benchmark.Configure(1000, 100, true);
  auto time = benchmark.Run(width, height, kind, enable_peer_access);
  std::cout << time << " ms" << std::endl;
}

#if HT_NVIDIA
TEST_CASE("Performance_hipMemcpyParam2DAsync_DeviceToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto width = GENERATE(2_KB, 4_KB, 8_KB);
  const auto height = width / 2;

  RunBenchmark(width, height, hipMemcpyDeviceToHost);
}
#endif

TEST_CASE("Performance_hipMemcpyParam2DAsync_HostToDevice") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto width = GENERATE(2_KB, 4_KB, 8_KB);
  const auto height = width / 2;

  RunBenchmark(width, height, hipMemcpyHostToDevice);
}

#if HT_NVIDIA
TEST_CASE("Performance_hipMemcpyParam2DAsync_HostToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto width = GENERATE(2_KB, 4_KB, 8_KB);
  const auto height = width / 2;

  RunBenchmark(width, height, hipMemcpyHostToHost);
}
#endif

TEST_CASE("Performance_hipMemcpyParam2DAsync_DeviceToDevice_DisablePeerAccess") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(2_KB, 4_KB, 8_KB);
  const auto height = width / 2;

  RunBenchmark(width, height, hipMemcpyDeviceToDevice);
}

TEST_CASE("Performance_hipMemcpyParam2DAsync_DeviceToDevice_EnablePeerAccess") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(2_KB, 4_KB, 8_KB);
  const auto height = width / 2;

  RunBenchmark(width, height, hipMemcpyDeviceToDevice, true);
}
