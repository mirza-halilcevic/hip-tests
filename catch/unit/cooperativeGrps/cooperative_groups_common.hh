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

#pragma once

#include "cpu_grid.h"

#include <optional>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

namespace {
#if HT_AMD
constexpr size_t kWarpSize = 64;
#else
constexpr size_t kWarpSize = 32;
#endif
}  // namespace

#define ASSERT_EQUAL(lhs, rhs) HIP_ASSERT(lhs == rhs)
#define ASSERT_LE(lhs, rhs) HIPASSERT(lhs <= rhs)
#define ASSERT_GE(lhs, rhs) HIPASSERT(lhs >= rhs)

constexpr int MaxGPUs = 8;

template <typename T> void compareResults(T* cpu, T* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(T); i++) {
    if (cpu[i] != gpu[i]) {
      INFO("Results do not match at index " << i);
      REQUIRE(cpu[i] == gpu[i]);
    }
  }
}

// Search if the sum exists in the expected results array
template <typename T> void verifyResults(T* hPtr, T* dPtr, int size) {
  int i = 0, j = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (hPtr[i] == dPtr[j]) {
        break;
      }
    }
    if (j == size) {
      INFO("Result verification failed!");
      REQUIRE(j != size);
    }
  }
}

inline bool operator==(const dim3& l, const dim3& r) {
  return l.x == r.x && l.y == r.y && l.z == r.z;
}

inline bool operator!=(const dim3& l, const dim3& r) { return !(l == r); }

template <typename T, typename F>
static inline void ArrayAllOf(const T* arr, uint32_t count, F value_gen) {
  for (auto i = 0u; i < count; ++i) {
    const std::optional<T> expected_val = value_gen(i);
    if (!expected_val.has_value()) continue;
    // Using require on every iteration leads to a noticeable performance loss on large arrays,
    // even when the require passes.
    if (arr[i] != expected_val.value()) {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] == expected_val.value());
    }
  }
}

__device__ inline unsigned int thread_rank_in_grid() {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  return block_rank_in_grid * block_size + thread_rank_in_block;
}

static __device__ void busy_wait(unsigned long long wait_period) {
  unsigned long long time_diff = 0;
  unsigned long long last_clock = clock64();
  while (time_diff < wait_period) {
    unsigned long long cur_clock = clock64();
    if (cur_clock > last_clock) {
      time_diff += (cur_clock - last_clock);
    }
    last_clock = cur_clock;
  }
}

static __device__ bool deactivate_thread(const uint64_t* const active_masks) {
  const auto warp = cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(), warpSize);
  const auto block = cooperative_groups::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  return !(active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank()));
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

inline dim3 GenerateThreadDimensions() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                            1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5};
  return GENERATE_COPY(
      dim3(1, 1, 1), dim3(props.maxThreadsDim[0], 1, 1), dim3(1, props.maxThreadsDim[1], 1),
      dim3(1, 1, props.maxThreadsDim[2]),
      map([max = props.maxThreadsDim[0], warp_size = props.warpSize](
              double i) { return dim3(std::min(static_cast<int>(i * warp_size), max), 1, 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[1], warp_size = props.warpSize](
              double i) { return dim3(1, std::min(static_cast<int>(i * warp_size), max), 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[2], warp_size = props.warpSize](
              double i) { return dim3(1, 1, std::min(static_cast<int>(i * warp_size), max)); },
          values(multipliers)),
      dim3(16, 8, 8), dim3(32, 32, 1), dim3(64, 8, 2), dim3(16, 16, 3), dim3(props.warpSize - 1, 3, 3),
      dim3(props.warpSize + 1, 3, 3));
}

inline dim3 GenerateBlockDimensions() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 2.0, 3.0, 4.0};
  return GENERATE_COPY(dim3(1, 1, 1),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(static_cast<int>(i * sm), 1, 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, static_cast<int>(i * sm), 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, 1, static_cast<int>(i * sm)); },
                           values(multipliers)),
                       dim3(5, 5, 5));
}

inline dim3 GenerateThreadDimensionsForShuffle() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.5, 0.9, 1.0, 1.5, 2.0};
  return GENERATE_COPY(
      dim3(1, 1, 1), dim3(props.maxThreadsDim[0], 1, 1), dim3(1, props.maxThreadsDim[1], 1),
      dim3(1, 1, props.maxThreadsDim[2]),
      map([max = props.maxThreadsDim[0], warp_size = props.warpSize](
              double i) { return dim3(std::min(static_cast<int>(i * warp_size), max), 1, 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[1], warp_size = props.warpSize](
              double i) { return dim3(1, std::min(static_cast<int>(i * warp_size), max), 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[2], warp_size = props.warpSize](
              double i) { return dim3(1, 1, std::min(static_cast<int>(i * warp_size), max)); },
          values(multipliers)),
      dim3(16, 8, 8), dim3(32, 32, 1), dim3(64, 8, 2), dim3(16, 16, 3), dim3(props.warpSize - 1, 3, 3),
      dim3(props.warpSize + 1, 3, 3));
}

inline dim3 GenerateBlockDimensionsForShuffle() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.5, 1.0};
  return GENERATE_COPY(dim3(1, 1, 1),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(static_cast<int>(i * sm), 1, 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, static_cast<int>(i * sm), 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, 1, static_cast<int>(i * sm)); },
                           values(multipliers)),
                       dim3(5, 5, 5));
}

template <typename Derived, typename T> class WarpTest {
 public:
  WarpTest() : warp_size_{get_warp_size()} {}

  void run() {
    const auto blocks = GenerateBlockDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    const auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    grid_ = CPUGrid(blocks, threads);

    const auto alloc_size = grid_.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    warps_in_block_ = (grid_.threads_in_block_count_ + warp_size_ - 1) / warp_size_;
    const auto warps_in_grid = warps_in_block_ * grid_.block_count_;
    LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                                warps_in_grid * sizeof(uint64_t));
    active_masks_.resize(warps_in_grid);
    std::generate(active_masks_.begin(), active_masks_.end(),
                  [] { return GenerateRandomInteger(0ul, std::numeric_limits<uint64_t>().max()); });

    HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks_.data(),
                        warps_in_grid * sizeof(uint64_t), hipMemcpyHostToDevice));
    cast_to_derived().launch_kernel(arr_dev.ptr(), active_masks_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    cast_to_derived().validate(arr.ptr());
  }

 private:
  int get_warp_size() const {
    int current_dev = -1;
    HIP_CHECK(hipGetDevice(&current_dev));
    int warp_size = 0u;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    return warp_size;
  }

  Derived& cast_to_derived() { return reinterpret_cast<Derived&>(*this); }

 protected:
  const int warp_size_;
  CPUGrid grid_;
  unsigned int warps_in_block_;
  std::vector<uint64_t> active_masks_;
};