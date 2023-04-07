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

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>

#include <hip_test_common.hh>

__global__ void rocrand_setup_kernel(rocrand_state_xorwow* states, uint64_t seed);

struct GraphEngineParams {
  void* input_generator;
  void* test_value_generator;
  void* reference_generator;
  void* validator;

  uint64_t input_sizeof;
  uint64_t test_value_sizeof;

  uint64_t base_value = 0;
  uint64_t batch_size;
};

template <bool brute_force = false> class GraphEngine {
 public:
  GraphEngine(const GraphEngineParams& params) : params_{params} {
    HIP_CHECK(hipMalloc(&rocrand_states_, sizeof(rocrand_state_xorwow)));

    const auto input_alloc_size = (brute_force ?: params_.batch_size) * params_.input_sizeof;
    const auto test_value_alloc_size = params_.batch_size * params_.test_value_sizeof;

    HIP_CHECK(hipMalloc(&input_dev_, input_alloc_size));
    HIP_CHECK(hipMalloc(&test_value_dev_, test_value_alloc_size));
    HIP_CHECK(hipMalloc(&n_dev_, sizeof(uint64_t)));

    HIP_CHECK(hipHostMalloc(&input_host_, input_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&test_value_host_, test_value_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&reference_host_, test_value_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&n_host_, sizeof(uint64_t), 0));

    HIP_CHECK(hipGraphCreate(&graph_, 0));
    PrepareNodeParams();
    AddNodes();
    AddDependencies();
    HIP_CHECK(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
  }

  GraphEngine(const GraphEngine&) = delete;
  GraphEngine& operator=(const GraphEngine&) = delete;

  GraphEngine(GraphEngine&&) = delete;
  GraphEngine& operator=(GraphEngine&&) = delete;

  ~GraphEngine() {
    hipError_t error;

    error = hipGraphExecDestroy(graph_exec_);
    if constexpr (brute_force) {
      error = hipGraphDestroyNode(input_htod_node_);
    } else {
      error = hipGraphDestroyNode(input_dtoh_node_);
      error = hipGraphDestroyNode(input_generator_node_);
    }
    error = hipGraphDestroyNode(n_htod_node_);
    error = hipGraphDestroyNode(test_value_dtoh_node_);
    error = hipGraphDestroyNode(validator_node_);
    error = hipGraphDestroyNode(reference_generator_node_);
    error = hipGraphDestroyNode(test_value_generator_node_);
    error = hipGraphDestroy(graph_);

    error = hipFree(n_host_);
    error = hipHostFree(reference_host_);
    error = hipHostFree(test_value_host_);
    error = hipHostFree(input_host_);

    error = hipFree(n_dev_);
    error = hipFree(test_value_dev_);
    error = hipFree(input_dev_);

    error = hipFree(rocrand_states_);
  }

  void Launch(uint64_t num_values, hipStream_t stream) {
    rocrand_setup_kernel<<<1, 1, 0, stream>>>(rocrand_states_, time(0));

    const auto orig_base_value = params_.base_value;
    const auto orig_batch_size = params_.batch_size;

    const auto tail_size = num_values % params_.batch_size;
    LaunchGraph(num_values - tail_size, stream);

    params_.batch_size = tail_size;
    LaunchGraph(tail_size, stream);

    params_.batch_size = orig_batch_size;
    params_.base_value = orig_base_value;
  }

 private:
  rocrand_state_xorwow* rocrand_states_;

  GraphEngineParams params_;

  void* input_dev_;
  void* test_value_dev_;
  uint64_t* n_dev_;

  void* input_host_;
  void* test_value_host_;
  void* reference_host_;
  uint64_t* n_host_;

  hipGraph_t graph_;
  hipGraphExec_t graph_exec_;

  hipKernelNodeParams input_generator_node_params_ = {0};
  hipKernelNodeParams test_value_generator_node_params_ = {0};
  hipHostNodeParams reference_generator_node_params_ = {0};
  hipHostNodeParams validator_node_params_ = {0};

  hipGraphNode_t input_generator_node_;
  hipGraphNode_t test_value_generator_node_;
  hipGraphNode_t reference_generator_node_;
  hipGraphNode_t validator_node_;
  hipGraphNode_t input_dtoh_node_;
  hipGraphNode_t input_htod_node_;
  hipGraphNode_t test_value_dtoh_node_;
  hipGraphNode_t n_htod_node_;

  std::string report_;

  std::array<void*, 3> input_generator_args_ = {&rocrand_states_, &n_dev_, &input_dev_};
  std::array<void*, 3> test_value_generator_args_ = {&n_dev_, &test_value_dev_, &input_dev_};
  std::array<void*, 3> reference_generator_args_ = {&n_host_, &reference_host_, &input_host_};
  std::array<void*, 5> validator_args_ = {&report_, &n_host_, &test_value_host_, &reference_host_,
                                          &input_host_};

  static void PrepareKernelNodeParams(hipKernelNodeParams& params, void* func) {
    params.func = func;
    int grid_size, block_size;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size, func, 0, 0));
    params.gridDim = dim3(grid_size);
    params.blockDim = dim3(block_size);
  }

  void PrepareNodeParams() {
    PrepareKernelNodeParams(input_generator_node_params_, params_.input_generator);
    input_generator_node_params_.kernelParams = input_generator_args_.data();

    PrepareKernelNodeParams(test_value_generator_node_params_, params_.test_value_generator);
    test_value_generator_node_params_.kernelParams = test_value_generator_args_.data();

    reference_generator_node_params_.fn =
        reinterpret_cast<hipHostFn_t>(params_.reference_generator);
    reference_generator_node_params_.userData = reference_generator_args_.data();

    validator_node_params_.fn = reinterpret_cast<hipHostFn_t>(params_.validator);
    validator_node_params_.userData = validator_args_.data();
  }

  void AddNodes() {
    if constexpr (brute_force) {
      HIP_CHECK(hipGraphAddMemcpyNode1D(&input_htod_node_, graph_, nullptr, 0, input_dev_,
                                        input_host_, params_.input_sizeof, hipMemcpyHostToDevice));
    } else {
      HIP_CHECK(hipGraphAddKernelNode(&input_generator_node_, graph_, nullptr, 0,
                                      &input_generator_node_params_));
      HIP_CHECK(hipGraphAddMemcpyNode1D(&input_dtoh_node_, graph_, nullptr, 0, input_host_,
                                        input_dev_, params_.batch_size * params_.input_sizeof,
                                        hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipGraphAddMemcpyNode1D(&n_htod_node_, graph_, nullptr, 0, n_dev_, n_host_,
                                      sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddKernelNode(&test_value_generator_node_, graph_, nullptr, 0,
                                    &test_value_generator_node_params_));
    HIP_CHECK(hipGraphAddHostNode(&reference_generator_node_, graph_, nullptr, 0,
                                  &reference_generator_node_params_));
    HIP_CHECK(hipGraphAddHostNode(&validator_node_, graph_, nullptr, 0, &validator_node_params_));
    HIP_CHECK(hipGraphAddMemcpyNode1D(
        &test_value_dtoh_node_, graph_, nullptr, 0, test_value_host_, test_value_dev_,
        params_.batch_size * params_.test_value_sizeof, hipMemcpyDeviceToHost));
  }

  void AddDependencies() {
    if constexpr (brute_force) {
      HIP_CHECK(hipGraphAddDependencies(graph_, &input_htod_node_, &test_value_generator_node_, 1));
    } else {
      HIP_CHECK(
          hipGraphAddDependencies(graph_, &input_generator_node_, &test_value_generator_node_, 1));
      HIP_CHECK(hipGraphAddDependencies(graph_, &input_generator_node_, &input_dtoh_node_, 1));
      HIP_CHECK(hipGraphAddDependencies(graph_, &input_dtoh_node_, &reference_generator_node_, 1));
      HIP_CHECK(hipGraphAddDependencies(graph_, &input_dtoh_node_, &validator_node_, 1));
    }
    HIP_CHECK(hipGraphAddDependencies(graph_, &n_htod_node_, &test_value_generator_node_, 1));
    HIP_CHECK(
        hipGraphAddDependencies(graph_, &test_value_generator_node_, &test_value_dtoh_node_, 1));
    HIP_CHECK(hipGraphAddDependencies(graph_, &test_value_dtoh_node_, &validator_node_, 1));
    HIP_CHECK(hipGraphAddDependencies(graph_, &reference_generator_node_, &validator_node_, 1));
  }

  void LaunchGraph(uint64_t num_values, hipStream_t stream) {
    for (uint64_t i = 0; i < num_values; i += params_.batch_size) {
      if constexpr (brute_force) {
        memcpy(input_host_, &params_.base_value, params_.input_sizeof);
      }
      n_host_[0] = params_.batch_size;

      HIP_CHECK(hipGraphLaunch(graph_exec_, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      if (!report_.empty()) {
        INFO(report_);
        REQUIRE(false);
      }

      params_.base_value += params_.batch_size;
    }
  }
};
