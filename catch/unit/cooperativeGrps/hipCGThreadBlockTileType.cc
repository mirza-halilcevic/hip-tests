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

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <bitset>

#include <resource_guards.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

/**
 * @addtogroup tiled_partition tiled_partition
 * @{
 * @ingroup coopGrpTest
 */

namespace cg = cooperative_groups;

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_size_getter(unsigned int* sizes) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    sizes[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).size();
  } else {
    sizes[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).size();
  }
}

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_thread_rank_getter(unsigned int* thread_ranks) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).thread_rank();
  } else {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).thread_rank();
  }
}

template <bool dynamic, size_t tile_size> void BlockTilePartitionGettersBasicTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto threads = GENERATE(dim3(53, 1, 1));
    auto blocks = GENERATE(dim3(3, 1, 1));
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

    thread_block_partition_size_getter<dynamic, tile_size><<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_partition_thread_rank_getter<dynamic, tile_size>
        <<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      if constexpr (!dynamic) {
        return tile_size;
      }

      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();

      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return tile_size - tail * (rank_in_block >= (partitions_in_block - 1) * tile_size);
    });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      return grid.thread_rank_in_block(i).value() % tile_size;
    });
  }
}

template <bool dynamic, size_t... tile_sizes> void BlockTilePartitionGettersBasicTest() {
  static_cast<void>((BlockTilePartitionGettersBasicTestImpl<dynamic, tile_sizes>(), ...));
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
TEST_CASE("Unit_Thread_Block_Tile_Getter_Positive_Basic") {
  BlockTilePartitionGettersBasicTest<false, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTilePartitionGettersBasicTest<false, 64>();
#endif
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
TEST_CASE("Unit_Thread_Block_Dynamic_Tile_Getter_Positive_Basic") {
  BlockTilePartitionGettersBasicTest<true, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTilePartitionGettersBasicTest<true, 64>();
#endif
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
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Up_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  TilePartitionShflUpTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  TilePartitionShflUpTest<TestType, 64>();
#endif
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
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Down_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  TilePartitionShflDownTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  TilePartitionShflDownTest<TestType, 64>();
#endif
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
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_XOR_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  TilePartitionShflXORTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  TilePartitionShflXORTest<TestType, 64>();
#endif
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
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  TilePartitionShflTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  TilePartitionShflTest<TestType, 64>();
#endif
}


static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
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

template <bool use_global, size_t tile_size, typename T>
__global__ void tiled_partition_sync_check(T* global_data, unsigned int* wait_modifiers) {
  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto tid = cg::this_grid().thread_rank();
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());

  const auto wait_modifier = wait_modifiers[tid];
  busy_wait(wait_modifier);
  data[tid] = partition.thread_rank();
  partition.sync();
  bool valid = true;
  for (auto i = 0; i < partition.size(); ++i) {
    const auto tile_base_idx = (tid / partition.size()) * partition.size();
    const auto expected = (partition.thread_rank() + i) % partition.size();
    const auto data_idx = tile_base_idx + expected;

    if (data_idx >= cg::this_grid().size()) {
      continue;
    }

    if (!(valid &= (data[tile_base_idx + expected] == expected))) {
      break;
    }
  }
  partition.sync();
  data[tid] = valid;
  if constexpr (!use_global) {
    global_data[tid] = data[tid];
  }
}

template <bool global_memory, typename T, size_t tile_size> void TiledPartitionSyncTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto randomized_run_count = GENERATE(range(0, 5));
    const auto threads = GENERATE_COPY(dim3(35, 1, 1));
    const auto blocks = dim3(1, 1, 1);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<unsigned int> wait_modifiers_dev(LinearAllocs::hipMalloc,
                                                      grid.thread_count_ * sizeof(unsigned int));
    LinearAllocGuard<unsigned int> wait_modifiers(LinearAllocs::hipHostMalloc,
                                                  grid.thread_count_ * sizeof(unsigned int));
    std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                  [] { return GenerateRandomInteger(0u, 1500u); });

    const auto shared_memory_size = global_memory ? 0u : alloc_size;
    HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                        grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

    tiled_partition_sync_check<global_memory, tile_size>
        <<<blocks, threads, shared_memory_size>>>(arr_dev.ptr(), wait_modifiers_dev.ptr());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(
        std::all_of(arr.ptr(), arr.ptr() + grid.thread_count_, [](unsigned int e) { return e; }));
  }
}

template <bool global_memory, typename T, size_t... tile_sizes> void TiledPartitionSyncTest() {
  static_cast<void>((TiledPartitionSyncTestImpl<global_memory, T, tile_sizes>(), ...));
}

TEMPLATE_TEST_CASE("Unit_Tiled_Partition_Sync_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  SECTION("Global memory") {
    TiledPartitionSyncTest<true, TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
    TiledPartitionSyncTest<true, TestType, 64>();
#endif
  }
  SECTION("Shared memory") {
    TiledPartitionSyncTest<false, TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
    TiledPartitionSyncTest<true, TestType, 64>();
#endif
  }
}

template <size_t warp_size> __device__ bool deactivate_thread(uint64_t* active_masks) {
  const auto warp = cg::tiled_partition<warp_size>(cg::this_thread_block());
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warp_size - 1) / warp_size;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warp.size();

  return !(active_masks[idx] & (1u << warp.thread_rank()));
}

template <size_t warp_size>
__global__ void coalesced_group_tiled_partition_size_getter(uint64_t* active_masks,
                                                            unsigned int tile_size,
                                                            unsigned int* sizes) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }
  sizes[thread_rank_in_grid()] = cg::tiled_partition(cg::coalesced_threads(), tile_size).size();
}

template <size_t warp_size>
__global__ void coalesced_group_tiled_partition_thread_rank_getter(uint64_t* active_masks,
                                                                   unsigned int tile_size,
                                                                   unsigned int* sizes) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }

  sizes[thread_rank_in_grid()] =
      cg::tiled_partition(cg::coalesced_threads(), tile_size).thread_rank();
}

TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Getters_Positive_Basic") {
  constexpr auto warp_size = 32u;
  const auto tile_size = GENERATE(2u, 4u, 8u, 16u, 32u);
  const auto x = GENERATE(range(1, 65));
  auto threads = dim3(x, 1, 1);
  const auto bx = GENERATE(range(1, 5));
  auto blocks = dim3(bx, 1, 1);
  INFO("Blocks: " << bx << ", threads: " << x);
  INFO("Tile size: " << tile_size);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
  LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  std::fill_n(active_masks.ptr(), warps_in_grid, static_cast<uint64_t>(0xFF1F00FF));
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_size_getter<32>
      <<<blocks, threads>>>(active_masks_dev.ptr(), tile_size, uint_arr_dev.ptr());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_thread_rank_getter<32>
      <<<blocks, threads>>>(active_masks_dev.ptr(), tile_size, uint_arr_dev.ptr());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_ +
      32 * TestContext::get().isNvidia();

  ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&](unsigned int idx) -> unsigned int {
    auto current_warp_mask = active_masks.ptr()[idx / warp_size];
    const auto i = grid.thread_rank_in_block(idx).value();
    if (~current_warp_mask & (1u << i % warp_size)) {
      return 0;
    }

    // Reset bits corresponding to threads that are inactive due to launch dimension
    // configuration
    // This only (potentially) applies to the final warp in a block
    const auto shift_amount = tail * (i / warp_size == warps_in_block - 1);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const uint64_t current_active_threads =
        std::bitset<sizeof(current_warp_mask) * 8>(current_warp_mask).count();

    const uint64_t active_up_to = current_warp_mask & ((1u << i % warp_size) - 1);
    const auto active_rank = std::bitset<sizeof(active_up_to) * 8>(active_up_to).count();

    const auto active_tiles = (current_active_threads + tile_size - 1) / tile_size;
    const auto active_tile_rank = active_rank / tile_size;
    const auto active_tail = active_tiles * tile_size - current_active_threads;
    return tile_size - active_tail * (active_tile_rank == active_tiles - 1);
  });

  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&](unsigned int idx) -> unsigned int {
    auto current_warp_mask = active_masks.ptr()[idx / warp_size];
    const auto i = grid.thread_rank_in_block(idx).value();
    if (~current_warp_mask & (1u << i % warp_size)) {
      return 0;
    }

    const uint64_t active_up_to = current_warp_mask & ((1u << i % warp_size) - 1);
    const auto active_rank = std::bitset<sizeof(active_up_to) * 8>(active_up_to).count();

    return active_rank % tile_size;
  });
}

template <typename T, size_t warp_size>
__global__ void coalesced_group_tiled_partition_shfl_up(uint64_t* active_masks, T* const out,
                                                        const unsigned int tile_size,
                                                        const unsigned int delta) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }
  const auto warp = cg::tiled_partition<warp_size>(cg::this_thread_block());
  T var = static_cast<T>(warp.thread_rank());

  const auto coalesced = cg::coalesced_threads();
  const auto tile = cg::tiled_partition(coalesced, tile_size);

  // printf("tid: %llu, var: %u\n", cg::this_grid().thread_rank(), tile.shfl_up(var, delta));

  out[thread_rank_in_grid()] = tile.shfl_up(var, delta);
}


TEST_CASE("Blahem") {
  constexpr auto warp_size = 32u;
  const auto tile_size = 8u;
  auto threads = dim3(31, 1, 1);
  auto blocks = dim3(1, 1, 1);
  CPUGrid grid(blocks, threads);
  const auto delta = 1u;
  using TestType = uint32_t;

  const auto alloc_size = grid.thread_count_ * sizeof(TestType);
  LinearAllocGuard<TestType> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<TestType> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + warp_size - 1) / warp_size;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  active_masks.ptr()[0] = 0xFEDCBA98;
  // active_masks.ptr()[1] = 0x80000002;
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_shfl_up<TestType, warp_size>
      <<<1, 32>>>(active_masks_dev.ptr(), uint_arr_dev.ptr(), tile_size, delta);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * warp_size - grid.threads_in_block_count_;
  std::array<unsigned int, warp_size> active_threads;

  for (auto i = 0u; i < warps_in_grid; ++i) {
    auto active_thread_count = 0u;
    active_threads.fill(0u);

    auto current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;
    for (auto t = 0u; t < warp_size; ++t) {
      if (current_warp_mask & (1u << t)) {
        active_threads[active_thread_count++] = t;
      }
    }
    std::cout << active_thread_count << std::endl;

    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t + delta;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; k++) {
        const auto global_thread_idx = i * warp_size + active_threads[k];
        const auto expected_val = active_threads[k - delta];
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        REQUIRE(actual_val == expected_val);
      }
    }
  }
}
