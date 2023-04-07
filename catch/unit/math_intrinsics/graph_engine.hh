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

#define ROCRAND_STATE rocrand_state_xorwow

__global__ void rocrand_setup_kernel(ROCRAND_STATE* states, uint64_t seed);

struct FailureReport {
  std::string msg;
};

struct GraphEngineParams {
  void* input_generator;
  void* test_value_generator;
  void* reference_generator;
  void* validator;

  uint64_t batch_size;
};

template <typename OutputType, typename InputType> class GraphEngine {
 public:
  GraphEngine(const GraphEngineParams& params) : params_{params} {
    const auto input_alloc_size = params_.batch_size * sizeof(InputType);
    const auto output_alloc_size = params_.batch_size * sizeof(OutputType);

    HIP_CHECK(hipMalloc(&input_dev_, input_alloc_size));
    HIP_CHECK(hipMalloc(&test_value_dev_, output_alloc_size));
    HIP_CHECK(hipMalloc(&n_dev_, sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&base_dev_, sizeof(uint64_t)));

    HIP_CHECK(hipHostMalloc(&input_host_, input_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&test_value_host_, output_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&reference_host_, output_alloc_size, 0));
    HIP_CHECK(hipHostMalloc(&n_host_, sizeof(uint64_t), 0));
    HIP_CHECK(hipHostMalloc(&base_host_, sizeof(uint64_t), 0));
    HIP_CHECK(hipHostMalloc(&report_, sizeof(FailureReport), 0));

    HIP_CHECK(hipGraphCreate(&graph_, 0));
    AddNodes();
    HIP_CHECK(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

    HIP_CHECK(hipMalloc(&rocrand_states_, sizeof(ROCRAND_STATE)));
  }

  GraphEngine(const GraphEngine&) = delete;
  GraphEngine& operator=(const GraphEngine&) = delete;

  GraphEngine(GraphEngine&&) = delete;
  GraphEngine& operator=(GraphEngine&&) = delete;

  ~GraphEngine() {
    hipError_t error;

    error = hipFree(input_dev_);
    error = hipFree(test_value_dev_);
    error = hipFree(n_dev_);
    error = hipFree(base_dev_);

    error = hipHostFree(input_host_);
    error = hipHostFree(test_value_host_);
    error = hipHostFree(reference_host_);
    error = hipHostFree(n_host_);
    error = hipHostFree(base_host_);
    error = hipHostFree(report_);

    error = hipGraphDestroy(graph_);

    error = hipGraphDestroyNode(input_generator_node_);
    error = hipGraphDestroyNode(test_value_generator_node_);
    error = hipGraphDestroyNode(reference_generator_node_);
    error = hipGraphDestroyNode(validator_node_);

    error = hipGraphDestroyNode(n_htod_node_);
    error = hipGraphDestroyNode(base_htod_node_);
    error = hipGraphDestroyNode(input_dtoh_node_);
    error = hipGraphDestroyNode(test_value_dtoh_node_);

    error = hipGraphExecDestroy(graph_exec_);

    error = hipFree(rocrand_states_);
  }

  void Launch(uint64_t begin, uint64_t end, hipStream_t stream) {
    rocrand_setup_kernel<<<1, 1, 0, stream>>>(rocrand_states_, time(0));

    const auto n = end - begin;
    const auto tail = n % params_.batch_size;

    Launch(begin, end - tail, params_.batch_size, stream);
    Launch(end - tail, end, tail, stream);
  }

 private:
  GraphEngineParams params_;

  void* input_dev_;
  void* test_value_dev_;
  uint64_t* n_dev_;
  uint64_t* base_dev_;

  void* input_host_;
  void* test_value_host_;
  void* reference_host_;
  uint64_t* n_host_;
  uint64_t* base_host_;

  hipGraph_t graph_;
  hipGraphExec_t graph_exec_;

  hipGraphNode_t input_generator_node_;
  hipGraphNode_t test_value_generator_node_;
  hipGraphNode_t reference_generator_node_;
  hipGraphNode_t validator_node_;
  hipGraphNode_t n_htod_node_;
  hipGraphNode_t base_htod_node_;
  hipGraphNode_t input_dtoh_node_;
  hipGraphNode_t test_value_dtoh_node_;

  hipKernelNodeParams input_generator_params_ = {0};
  hipKernelNodeParams test_value_generator_params_ = {0};
  hipHostNodeParams reference_generator_params_ = {0};
  hipHostNodeParams validator_params_ = {0};

  ROCRAND_STATE* rocrand_states_;
  std::array<void*, 4> input_generator_args_ = {&rocrand_states_, &base_dev_, &n_dev_, &input_dev_};

  std::array<void*, 3> test_value_generator_args_ = {&n_dev_, &test_value_dev_, &input_dev_};

  std::array<void*, 3> reference_generator_args_ = {&n_host_, &reference_host_, &input_host_};

  FailureReport* report_;
  std::array<void*, 5> validator_args_ = {&report_, &n_host_, &test_value_host_, &reference_host_,
                                          &input_host_};

  static void PrepareKernelNodeParams(hipKernelNodeParams& params, void* func) {}

  void PrepareNodeParams() {
    int grid_size, block_size;

    input_generator_params_.func = params_.input_generator;
    HIP_CHECK(
        hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size, params_.input_generator, 0, 0));
    input_generator_params_.gridDim = dim3(grid_size);
    input_generator_params_.blockDim = dim3(block_size);
    input_generator_params_.kernelParams = input_generator_args_.data();

    test_value_generator_params_.func = params_.test_value_generator;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                params_.test_value_generator, 0, 0));
    test_value_generator_params_.gridDim = dim3(grid_size);
    test_value_generator_params_.blockDim = dim3(block_size);
    test_value_generator_params_.kernelParams = test_value_generator_args_.data();

    reference_generator_params_.fn = reinterpret_cast<hipHostFn_t>(params_.reference_generator);
    reference_generator_params_.userData = reference_generator_args_.data();

    validator_params_.fn = reinterpret_cast<hipHostFn_t>(params_.validator);
    validator_params_.userData = validator_args_.data();
  }

  void AddNodes() {
    PrepareNodeParams();

    std::vector<hipGraphNode_t> node_deps;

    // root -> n_htod
    HIP_CHECK(hipGraphAddMemcpyNode1D(&n_htod_node_, graph_, nullptr, 0, n_dev_, n_host_,
                                      sizeof(uint64_t), hipMemcpyHostToDevice));

    // root -> base_htod
    HIP_CHECK(hipGraphAddMemcpyNode1D(&base_htod_node_, graph_, nullptr, 0, base_dev_, base_host_,
                                      sizeof(uint64_t), hipMemcpyHostToDevice));

    // n_htod + base_htod -> input_generator
    node_deps.push_back(n_htod_node_);
    node_deps.push_back(base_htod_node_);
    HIP_CHECK(hipGraphAddKernelNode(&input_generator_node_, graph_, node_deps.data(),
                                    node_deps.size(), &input_generator_params_));
    node_deps.clear();

    // input_generator -> input_dtoh
    node_deps.push_back(input_generator_node_);
    HIP_CHECK(hipGraphAddMemcpyNode1D(
        &input_dtoh_node_, graph_, node_deps.data(), node_deps.size(), input_host_, input_dev_,
        params_.batch_size * sizeof(InputType), hipMemcpyDeviceToHost));
    node_deps.clear();

    // input_generator -> test_value_generator
    node_deps.push_back(input_dtoh_node_);
    HIP_CHECK(hipGraphAddKernelNode(&test_value_generator_node_, graph_, node_deps.data(),
                                    node_deps.size(), &test_value_generator_params_));
    node_deps.clear();

    // input_dtoh -> reference_generator
    node_deps.push_back(input_dtoh_node_);
    HIP_CHECK(hipGraphAddHostNode(&reference_generator_node_, graph_, node_deps.data(),
                                  node_deps.size(), &reference_generator_params_));
    node_deps.clear();

    // test_value_generator -> test_value_dtoh
    node_deps.push_back(test_value_generator_node_);
    node_deps.push_back(reference_generator_node_);
    HIP_CHECK(hipGraphAddMemcpyNode1D(
        &test_value_dtoh_node_, graph_, node_deps.data(), node_deps.size(), test_value_host_,
        test_value_dev_, params_.batch_size * sizeof(OutputType), hipMemcpyDeviceToHost));
    node_deps.clear();

    // test_value_dtoh + reference_generator -> validator
    node_deps.push_back(test_value_dtoh_node_);
    node_deps.push_back(reference_generator_node_);
    HIP_CHECK(hipGraphAddHostNode(&validator_node_, graph_, node_deps.data(), node_deps.size(),
                                  &validator_params_));
    node_deps.clear();
  }

  void Launch(uint64_t begin, uint64_t end, uint64_t batch_size, hipStream_t stream) {
    n_host_[0] = batch_size;

    for (; begin < end; begin += batch_size) {
      base_host_[0] = begin;

      HIP_CHECK(hipGraphLaunch(graph_exec_, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      if (!report_->msg.empty()) {
        INFO(report_->msg);
        REQUIRE(false);
      }
    }
  }
};
