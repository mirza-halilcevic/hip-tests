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

#include <hip/hip_cooperative_groups.h>

#include <resource_guards.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

#ifdef _CG_HAS_FP16_COLLECTIVE
#define FP16 , __half
#else
#define FP16 
#endif

/**
 * @addtogroup tiled_partition tiled_partition
 * @{
 * @ingroup coopGrpTest
 */

namespace cg = cooperative_groups;

template <unsigned int tile_size, bool dynamic = false>
__global__ void thread_block_partition_size_getter(unsigned int* sizes) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    sizes[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).size();
  } else {
    sizes[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).size();
  }
}

template <unsigned int tile_size, bool dynamic = false>
__global__ void thread_block_partition_thread_rank_getter(unsigned int* thread_ranks) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).thread_rank();
  } else {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).thread_rank();
  }
}

template <size_t tile_size, bool dynamic = false> void BlockTilePartitionGettersBasicTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(2, 1, 1));
    auto blocks = GENERATE(dim3(3, 1, 1));
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

    thread_block_partition_size_getter<tile_size><<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_partition_thread_rank_getter<tile_size><<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [](unsigned int) { return tile_size; });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      return grid.thread_rank_in_block(i).value() % tile_size;
    });
  }
}

template <bool dynamic, size_t... tile_sizes> void BlockTilePartitionGettersBasicTest() {
  static_cast<void>((BlockTilePartitionGettersBasicTestImpl<tile_sizes, dynamic>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Creates tile partitions for each of the valid sizes{2, 4, 8, 16, 32} and writes the return
 * values of size and thread_rank member functions to an output array that is validated on the host
 * side.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Thread_Block_Tile_Getter_Positive_Basic") {
  BlockTilePartitionGettersBasicTest<false, 2, 4, 8, 16, 32>();
}

/**
 * Test Description
 * ------------------------
 *    - Creates tile partitions for each of the valid sizes{2, 4, 8, 16, 32} via the dynamic tiled
 * partition api and writes the return values of size and thread_rank member functions to an output
 * array that is validated on host.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Thread_Block_Dynamic_Tile_Getter_Positive_Basic") {
  BlockTilePartitionGettersBasicTest<true, 2, 4, 8, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_partition_shfl_up(T* const out, const unsigned int delta) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_up(var, delta);
}

template <typename T, size_t tile_size> void TilePartitionShflUpTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(3, 1, 1), dim3(57, 2, 8));
    auto blocks = GENERATE(dim3(2, 1, 1), dim3(5, 5, 5));
    auto delta = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Delta: " << delta);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_partition_shfl_up<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const int rank_in_partition = grid.thread_rank_in_block(i).value() % tile_size;
      const int target = rank_in_partition - delta;
      return target < 0 ? rank_in_partition : target;
    });
  }
}

template <typename T, size_t... tile_sizes> void TilePartitionShflUpTest() {
  static_cast<void>((TilePartitionShflUpTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle up behavior of tiled groups of all valid sizes{2, 4, 8, 16, 32} for
 * delta values of [0, tile size). The test is run for all overloads of shfl_up.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
// Add FP16 type tests if supported
TEMPLATE_TEST_CASE("Thread_Block_Tile_Shfl_Up_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double FP16) {
  TilePartitionShflUpTest<TestType, 2, 4, 8, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_partition_shfl_down(T* const out, const unsigned int delta) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_down(var, delta);
}

template <typename T, size_t tile_size> void TilePartitionShflDownTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(3, 1, 1));
    auto blocks = GENERATE(dim3(2, 1, 1));
    auto delta = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Delta: " << delta);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_partition_shfl_down<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const auto rank_in_group = rank_in_block % tile_size;
      const auto target = rank_in_group + delta;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target < tile_size ? target : rank_in_group;
      } else {
        // If the number of threads in a block is not an integer multiple of tile_size, the
        // final(tail end) tile will contain inactive threads.
        // Shuffling from an inactive thread returns an undefined value, accordingly threads that
        // shuffle from one must be skipped
        const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
        return target < tile_size - tail ? std::optional(target) : std::nullopt;
      }
    });
  }
}

template <typename T, size_t... tile_sizes> void TilePartitionShflDownTest() {
  static_cast<void>((TilePartitionShflDownTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle down behavior of tiled groups of all valid sizes{2, 4, 8, 16, 32} for
 * delta values of [0, tile size). The test is run for all overloads of shfl_down.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
// Add FP16 type tests if supported
TEMPLATE_TEST_CASE("Thread_Block_Tile_Shfl_Down_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double FP16) {
  TilePartitionShflDownTest<TestType, 2, 4, 8, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_partition_shfl_xor(T* const out, const unsigned mask) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_xor(var, mask);
}

template <typename T, size_t tile_size> void TilePartitionShflXORTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(3, 1, 1));
    auto blocks = GENERATE(dim3(2, 1, 1));
    const auto mask = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Mask: 0x" << std::hex << mask);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_partition_shfl_xor<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), mask);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [mask, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = rank_in_partition ^ mask;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void TilePartitionShflXORTest() {
  static_cast<void>((TilePartitionShflXORTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle xor behavior of tiled groups of all valid sizes{2, 4, 8, 16, 32} for
 * mask values of [0, tile size). The test is run for all overloads of shfl_xor.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
// Add FP16 type tests if supported
TEMPLATE_TEST_CASE("Thread_Block_Tile_Shfl_XOR_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double FP16) {
  TilePartitionShflXORTest<TestType, 2, 4, 8, 16, 32>();
}


template <typename T, size_t tile_size>
__global__ void block_tile_partition_shfl(T* const out, uint8_t* target_lanes) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl(var, target_lanes[partition.thread_rank()]);
}

template <typename T, size_t tile_size> void TilePartitionShflTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(3, 1, 1));
    auto blocks = GENERATE(dim3(2, 1, 1));
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc,
                                               tile_size * sizeof(uint8_t));
    LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc,
                                           tile_size * sizeof(uint8_t));
    // Generate a couple different combinations for target lanes
    for (auto i = 0u; i < tile_size; ++i) {
      target_lanes.ptr()[i] = tile_size - 1 - i;
    }

    HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(), tile_size * sizeof(uint8_t),
                        hipMemcpyHostToDevice));
    block_tile_partition_shfl<T, tile_size>
        <<<blocks, threads>>>(arr_dev.ptr(), target_lanes_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [&target_lanes, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = target_lanes.ptr()[rank_in_partition] % tile_size;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void TilePartitionShflTest() {
  static_cast<void>((TilePartitionShflTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle behavior of tiled groups of all valid sizes{2, 4, 8, 16, 32} for
 * ... . The test is run for all overloads of shfl.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/hipCGThreadBlockTileType.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Thread_Block_Tile_Shfl_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double FP16) {
  TilePartitionShflTest<TestType, 2, 4, 8, 16, 32>();
}