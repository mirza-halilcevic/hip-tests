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

#include <functional>

#include "vulkan_test.hh"
#include "wait_semaphore_common.hh"
#include "graph_tests_common.hh"

TEST_CASE("Unit_hipGraphAddExternalSemaphoresWaitNode_Positive_Basic") {
  WaitExternalSemaphoreCommon(GraphExtSemaphoreWaitWrapper<>);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA
TEST_CASE("Unit_hipGraphAddExternalSemaphoresWaitNode_Vulkan_Positive_Timeline_Semaphore") {
  WaitExternalTimelineSemaphoreCommon(GraphExtSemaphoreWaitWrapper<>);
}
#endif

TEST_CASE("Unit_hipGraphAddExternalSemaphoresWaitNode_Vulkan_Positive_Multiple_Semaphores") {
  WaitExternalMultipleSemaphoresCommon(GraphExtSemaphoreWaitWrapper<>);
}

TEST_CASE("Unit_hipGraphAddExternalSemaphoresWaitNode_Vulkan_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;
  auto hip_ext_semaphore = ImportBinarySemaphore(vkt);

  hipExternalSemaphoreWaitNodeParams node_params = {};
  node_params.extSemArray = &hip_ext_semaphore;
  node_params.paramsArray = &wait_params;
  node_params.numExtSems = 1;

  GraphAddNodeCommonNegativeTests(
      std::bind(hipGraphAddExternalSemaphoresWaitNode, _1, _2, _3, _4, &node_params), graph);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  HIP_CHECK(hipGraphDestroy(graph));
}
