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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <functional>
#include <vector>

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "graph_tests_common.hh"

TEMPLATE_TEST_CASE("Unit_hipGraphAddMemsetNode_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  const size_t width = GENERATE(1, 64, kPageSize / sizeof(TestType) + 1);
  const size_t height = GENERATE(1, 2, 1024);

  LinearAllocGuard2D<TestType> alloc(width, height);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node = nullptr;
  constexpr TestType set_value = 42;
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(TestType);
  params.width = width;
  params.height = height;
  params.pitch = alloc.pitch();
  params.value = set_value;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  LinearAllocGuard<TestType> buffer(LinearAllocs::hipHostMalloc, width * sizeof(TestType) * height);
  HIP_CHECK(hipMemcpy2D(buffer.ptr(), width * sizeof(TestType), alloc.ptr(), alloc.pitch(),
                        width * sizeof(TestType), height, hipMemcpyDeviceToHost));
  ArrayFindIfNot(buffer.ptr(), set_value, width * height);
}

TEST_CASE("Unit_hipGraphAddMemsetNode_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 1;
  params.value = 42;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddMemsetNode, _1, _2, _3, _4, &params), graph);

  hipGraphNode_t node = nullptr;

  SECTION("pMemsetParams == nullptr") {
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, nullptr), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.dst == nullptr") {
    params.dst = nullptr;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.elementSize != 1, 2, 4") {
    params.elementSize = GENERATE(0, 3, 5);
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.width == 0") {
    params.width = 0;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.width > allocation size") {
    params.width = 5;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.height == 0") {
    params.height = 0;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.pitch < width when height > 1") {
    params.width = 2;
    params.height = 2;
    params.pitch = 1 * params.elementSize;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  SECTION("pMemsetParams.pitch * height > allocation size") {
    params.width = 2;
    params.height = 2;
    params.pitch = 3 * params.elementSize;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}