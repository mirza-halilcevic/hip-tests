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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

std::string to_string(const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::malloc:
      return "host pageable";
    case LinearAllocs::hipHostMalloc:
      return "host pinned";
    case LinearAllocs::hipMalloc:
      return "device malloc";
    case LinearAllocs::hipMallocManaged:
      return "managed";
    default:
      return "unknown alloc type";
  }
}

template <typename T, bool use_shared_mem>
__global__ void atomicExchKernel(T* const global_mem, T* const old_vals) {
  __shared__ T shared_mem;

  const auto tid = cg::this_grid().thread_rank();

  T* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = atomicExch(mem, static_cast<T>(tid + 1));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType, bool use_shared_mem>
void AtomicExchSameAddressTest(const dim3 blocks, const dim3 threads,
                               const LinearAllocs alloc_type) {
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType));

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count + 1);


  HIP_CHECK(hipMemset(mem_dev.ptr(), 0, sizeof(TestType)));
  atomicExchKernel<TestType, use_shared_mem>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, mem_dev.ptr(), sizeof(TestType),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Every thread will exchange its grid-wide linear id into a target location within mem_dev,
  // receiving back the value previously present therein. This previous value is written to
  // old_vals_dev.
  // old_vals_dev will not contain values that the final scheduled warp exchanged into mem_dev, but
  // mem_dev obviously will.
  // Given that mem_dev initially contains values in the range [0, width) and that the maximum value
  // the final thread shall write is thread_count + width - 1, presuming correct operation of
  // atomicExch, the union of mem_dev and old_vals_dev shall contain values in the range
  //[0, thread_count + width)
  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Same_Address", "", int, unsigned int,
                   unsigned long long, float) {
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(20));
    using LA = LinearAllocs;
    const auto allocation_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    AtomicExchSameAddressTest<TestType, false>(blocks, threads, allocation_type);
  }

  SECTION("Shared memory") {
    const auto blocks = GENERATE(dim3(1));
    AtomicExchSameAddressTest<TestType, true>(blocks, threads, LinearAllocs::hipMalloc);
  }
}

enum class AtomicExchScopes { device, system };

template <typename T, AtomicExchScopes scope> __device__ T perform_atomic_exch(T* address, T val) {
  if constexpr (scope == AtomicExchScopes::device) {
    return atomicExch(address, val);
  } else if (scope == AtomicExchScopes::system) {
    return atomicExch_system(address, val);
  }
}


template <typename T>
__device__ T* pitched_offset(T* const ptr, const unsigned int pitch, const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<T*>(byte_ptr + idx * pitch);
}

template <typename T, bool use_shared_mem>
__global__ void atomicExchMultiDestKernel(T* const global_mem, T* const old_vals,
                                          const unsigned int width, const unsigned pitch,
                                          const T base_val = 0) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cg::this_grid().thread_rank();

  T* const mem = use_shared_mem ? reinterpret_cast<T*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) {
      const auto target = pitched_offset(mem, pitch, tid);
      *target = *pitched_offset(global_mem, pitch, tid);
    };
    __syncthreads();
  }

  old_vals[tid] =
      atomicExch(pitched_offset(mem, pitch, tid % width), base_val + static_cast<T>(tid + width));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = pitched_offset(global_mem, pitch, tid);
      *target = *pitched_offset(mem, pitch, tid);
    };
  }
}

template <typename TestType, bool use_shared_mem, unsigned int kernel_count = 1u>
void AtomicExchMultiDestWithScatter(const dim3 blocks, const dim3 threads,
                                    const LinearAllocs alloc_type, const unsigned int width,
                                    const unsigned int pitch) {
  static_assert(!(kernel_count > 1 && use_shared_mem),
                "Shared memory should not be used for a multiple kernel launch");

  if constexpr (kernel_count > 1) {
    int concurrent_kernels = 0;
    HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, 0));
    if (!concurrent_kernels) {
      HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernels");
      return;
    }
  }

  const auto mem_alloc_size = width * pitch;
  LinearAllocGuard<TestType> mem_dev(alloc_type, mem_alloc_size);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType) * kernel_count;
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count * kernel_count + width);
  std::iota(old_vals.begin(), old_vals.begin() + width, 0);

  struct CreatedStream {
    auto stream() { return stream_.stream(); }
    StreamGuard stream_{Streams::created};
  };
  std::array<CreatedStream, kernel_count> streams;

  HIP_CHECK(hipMemcpy2D(mem_dev.ptr(), pitch, old_vals.data(), sizeof(TestType), sizeof(TestType),
                        width, hipMemcpyHostToDevice));
  const auto shared_mem_size = use_shared_mem ? mem_alloc_size : 0u;
  for (auto i = 0; i < kernel_count; ++i) {
    atomicExchMultiDestKernel<TestType, use_shared_mem>
        <<<blocks, threads, shared_mem_size, streams[i].stream()>>>(
            mem_dev.ptr(), old_vals_dev.ptr() + thread_count * i, width, pitch, thread_count * i);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy2D(old_vals.data() + kernel_count * thread_count, sizeof(TestType),
                        mem_dev.ptr(), pitch, sizeof(TestType), width, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

template <typename TestType>
void AtomicExchSingleDeviceSingleKernelTest(const unsigned int width, const unsigned int pitch) {
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(40));
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        AtomicExchMultiDestWithScatter<TestType, false, 1>(blocks, threads, LinearAllocs::hipMalloc,
                                                           width, pitch);
      }
    }
  }

  SECTION("Shared memory") {
    const auto blocks = GENERATE(dim3(1));
    AtomicExchMultiDestWithScatter<TestType, true, 1>(blocks, threads, LinearAllocs::hipMalloc,
                                                      width, pitch);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Same_Address_Runtime", "", int, unsigned int,
                   unsigned long long, float) {
  AtomicExchSingleDeviceSingleKernelTest<TestType>(1, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Adjacent_Addresses", "", int, unsigned int,
                   unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  AtomicExchSingleDeviceSingleKernelTest<TestType>(warp_size, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Scattered_Addresses", "", int, unsigned int,
                   unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  AtomicExchSingleDeviceSingleKernelTest<TestType>(warp_size, cache_line_size);
}


template <typename TestType, unsigned int kernel_count>
void AtomicExchSingleDeviceMultipleKernelTest(const unsigned int width, const unsigned int pitch) {
  const auto blocks = GENERATE(dim3(40));
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      AtomicExchMultiDestWithScatter<TestType, false, kernel_count>(
          blocks, threads, LinearAllocs::hipMalloc, width, pitch);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Same_Address_Runtime_Multi_Kernel", "", int,
                   unsigned int, unsigned long long, float) {
  AtomicExchSingleDeviceMultipleKernelTest<TestType, 2>(1, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Adjacent_Addresses_Multi_Kernel", "", int,
                   unsigned int, unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  AtomicExchSingleDeviceMultipleKernelTest<TestType, 2>(warp_size, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Scattered_Addresses_Multi_Kernel", "", int,
                   unsigned int, unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  AtomicExchSingleDeviceMultipleKernelTest<TestType, 2>(warp_size, cache_line_size);
}

template <typename T>
__global__ void atomicExchSystemKernel(T* const global_mem, T* const old_vals,
                                       const unsigned int width, const unsigned pitch,
                                       const T base_val = 0) {
  T* const mem = global_mem;

  const auto tid = cg::this_grid().thread_rank();

  const auto new_value = base_val + static_cast<T>(tid + width);
  const auto old_value = atomicExch_system(pitched_offset(mem, pitch, tid % width), new_value);
  // printf("tid: %u, old_value: %d, new_value: %d\n", tid, old_value, new_value);
  old_vals[tid] = old_value;
}

template <typename TestType, bool use_shared_mem, unsigned int num_devices,
          unsigned int kernel_count = 1u>
void Foo(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type,
         const unsigned int width, const unsigned int pitch) {
  static_assert(!(kernel_count > 1 && use_shared_mem),
                "Shared memory should not be used for a multiple kernel launch");

  if constexpr (num_devices > 1) {
    if (HipTest::getDeviceCount() < num_devices) {
      std::string msg = std::to_string(num_devices) + " devices are required";
      HipTest::HIP_SKIP_TEST(msg.c_str());
      return;
    }
  }

  if constexpr (kernel_count > 1) {
    for (auto i = 0u; i < num_devices; ++i) {
      int concurrent_kernels = 0;
      HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, i));
      if (!concurrent_kernels) {
        HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
        return;
      }
    }
  }

  const auto mem_alloc_size = width * pitch;
  LinearAllocGuard<TestType> mem_dev(alloc_type, mem_alloc_size);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType) * kernel_count;
  std::vector<LinearAllocGuard<TestType>> old_vals_devs;
  for (auto i = 0u; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    old_vals_devs.emplace_back(LinearAllocs::hipMalloc, old_vals_alloc_size);
  }

  StreamGuard streams[num_devices][kernel_count];
  for (auto i = 0; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (auto j = 0; j < kernel_count; ++j) {
      streams[i][j] = StreamGuard(Streams::created);
    }
  }

  std::vector<TestType> old_vals(num_devices * thread_count * kernel_count + width);
  std::iota(old_vals.begin(), old_vals.begin() + width, 0);

  HIP_CHECK(hipMemcpy2D(mem_dev.ptr(), pitch, old_vals.data(), sizeof(TestType), sizeof(TestType),
                        width, hipMemcpyDefault));

  for (auto i = 0; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    const auto device_offset = thread_count * kernel_count * i;
    for (auto j = 0; j < kernel_count; ++j) {
      atomicExchSystemKernel<TestType><<<blocks, threads, 0, streams[i][j].stream()>>>(
          mem_dev.ptr(), old_vals_devs[i].ptr() + thread_count * j, width, pitch,
          device_offset + thread_count * j);
      HIP_CHECK(hipGetLastError());
    }
  }

  for (auto i = 0u; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<TestType> scratch(kernel_count * thread_count);
    HIP_CHECK(hipMemcpy(scratch.data(), old_vals_devs[i].ptr(), old_vals_alloc_size,
                        hipMemcpyDeviceToHost));
    std::cout << i << std::endl;
    for (auto e : scratch) {
      std::cout << e << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::vector<TestType> bla(width);
  HIP_CHECK(hipMemcpy2D(bla.data(), sizeof(TestType), mem_dev.ptr(), pitch, sizeof(TestType), width,
                        hipMemcpyDeviceToHost));
  for (auto e : bla) {
    std::cout << e << ", ";
  }
  std::cout << std::endl << std::endl;

  for (auto i = 0u; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    const auto device_offset = thread_count * kernel_count * i;
    HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                        old_vals_alloc_size, hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipMemcpy2D(old_vals.data() + num_devices * kernel_count * thread_count,
                        sizeof(TestType), mem_dev.ptr(), pitch, sizeof(TestType), width,
                        hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  std::sort(old_vals.begin(), old_vals.end());
  for (auto e : old_vals) {
    std::cout << e << ", ";
  }
  std::cout << std::endl;
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

template <typename TestType, unsigned int num_devices, unsigned int kernel_count>
void Bar(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type,
         const unsigned int width, const unsigned int pitch) {
  if constexpr (num_devices > 1) {
    if (HipTest::getDeviceCount() < num_devices) {
      std::string msg = std::to_string(num_devices) + " devices are required";
      HipTest::HIP_SKIP_TEST(msg.c_str());
      return;
    }
  }

  if constexpr (kernel_count > 1) {
    for (auto i = 0u; i < num_devices; ++i) {
      int concurrent_kernels = 0;
      HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, i));
      if (!concurrent_kernels) {
        HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
        return;
      }
    }
  }

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;

  const auto old_vals_alloc_size = kernel_count * thread_count * sizeof(TestType);
  std::vector<LinearAllocGuard<TestType>> old_vals_devs;
  std::vector<StreamGuard> streams;
  for (auto i = 0; i < num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    old_vals_devs.emplace_back(LinearAllocs::hipMalloc, old_vals_alloc_size);
    for (auto j = 0; j < kernel_count; ++j) {
      streams.emplace_back(Streams::created);
    }
  }

  const auto mem_alloc_size = width * pitch;
  LinearAllocGuard<TestType> mem_dev(alloc_type, mem_alloc_size);

  std::vector<TestType> old_vals(num_devices * kernel_count * thread_count + width);
  std::iota(old_vals.begin(), old_vals.begin() + width, 0);

  HIP_CHECK(hipMemcpy2D(mem_dev.ptr(), pitch, old_vals.data(), sizeof(TestType), sizeof(TestType),
                        width, hipMemcpyHostToDevice));

  for (auto i = 0u; i < num_devices; ++i) {
    const auto device_offset = i * kernel_count * thread_count;
    for (auto j = 0u; j < kernel_count; ++j) {
      const auto& stream = streams[i * kernel_count + j].stream();
      const auto kern_offset = j * thread_count;
      const auto old_vals = old_vals_devs[i].ptr() + kern_offset;
      atomicExchSystemKernel<TestType><<<blocks, threads, 0, stream>>>(
          mem_dev.ptr(), old_vals, width, pitch, device_offset + kern_offset);
    }
  }

  for (auto i = 0u; i < num_devices; ++i) {
    const auto device_offset = i * kernel_count * thread_count;
    HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                        old_vals_alloc_size, hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipMemcpy2D(old_vals.data() + num_devices * kernel_count * thread_count,
                        sizeof(TestType), mem_dev.ptr(), pitch, sizeof(TestType), width,
                        hipMemcpyDeviceToHost));

  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}


TEMPLATE_TEST_CASE("Bar", "", int, unsigned int, unsigned long long, float) {
  const auto blocks = GENERATE(dim3(20));
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(1024));

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMallocManaged, LA::hipMallocManaged, LA::mallocAndRegister}) {
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      Bar<TestType, 3, 3>(blocks, threads, alloc_type, 32, 128);
    }
  }
}
