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

/**
Testcase Scenarios of hipGraphExecMemcpyNodeSetParamsToSymbol API:
Functional :
1) Allocate global symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
2) Allocate const symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
Negative :
1) Pass hGraphExec as nullptr and check if api returns error.
2) Pass GraphNode as nullptr and check if api returns error.
3) Pass symbol ptr as nullptr, api expected to return error code.
4) Pass source ptr as nullptr, api expected to return error code.
5) Pass count as zero, api expected to return error code.
6) Pass offset+count greater than allocated size, api expected to return error code.
7) Pass same symbol pointer as source ptr and destination ptr, api expected to return error code.
8) Pass Pass both dstn ptr and source ptr as 2 different symbol ptr, api expected to return error code.
9) Copy from device ptr to host ptr but pass kind as different, api expected to return error code.
10) Check with other graph node but pass same graphExec, api expected to return error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <limits>
#define SIZE 256

__device__ int globalIn[SIZE], globalOut[SIZE];
__device__ __constant__ int globalConst[SIZE];

__global__ void MemcpyToSymbolExecKernel(int* B_d) {
  for (int i = 0 ; i < SIZE; i++) {
      B_d[i] = globalIn[i];
  }
}

__global__ void MemcpyToConstSymbolExecKernel(int* B_d) {
  for (int i = 0 ; i < SIZE; i++) {
      B_d[i] = globalConst[i];
  }
}

/* This testcase verifies negative scenarios of
   hipGraphExecMemcpyNodeSetParamsToSymbol API */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipError_t ret;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyH2D;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D);

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            HIP_SYMBOL(globalIn),
                                            C_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  HIP_CHECK(hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                            HIP_SYMBOL(globalIn), A_d,
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(nullptr,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  B_d, Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass GraphNode as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  nullptr,
                                                  HIP_SYMBOL(globalIn),
                                                  B_d, Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass symbol ptr as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  nullptr,
                                                  B_d, Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidSymbol == ret);
  }
  SECTION("Pass source ptr as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  nullptr, Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count as zero") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  B_d, 0, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass offset+count greater than allocated size") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  B_d, Nbytes, 10,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass same symbol pointer as source ptr and destination ptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  HIP_SYMBOL(globalIn),
                                                  Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass both dstn ptr and source ptr as 2 different symbol ptr") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  HIP_SYMBOL(globalOut),
                                                  Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Copy from device ptr to host ptr but pass kind as different") {
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode,
                                                  HIP_SYMBOL(globalIn),
                                                  B_h,
                                                  Nbytes, 0,
                                                  hipMemcpyDeviceToHost);
    REQUIRE(hipSuccess != ret);
  }
  SECTION("Check with other graph node but pass same graphExec") {
    hipGraph_t graph1;
    hipGraphNode_t memcpyToSymbolNode1{};
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode1, graph1,
                                            nullptr,
                                            0,
                                            HIP_SYMBOL(globalOut),
                                            A_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
    ret = hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                                  memcpyToSymbolNode1,
                                                  HIP_SYMBOL(globalOut),
                                                  HIP_SYMBOL(globalIn),
                                                  Nbytes, 0,
                                                  hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

static
void hipGraphExecMemcpyNodeSetParamsToSymbol_GlobalMem(bool useConstVar) {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, SIZE);
  hipGraphNode_t memcpytosymbolkernel, memcpyD2H_B;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  if (useConstVar) {
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            HIP_SYMBOL(globalConst),
                                            C_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  } else {
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            HIP_SYMBOL(globalIn),
                                            C_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  }
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  // Adding Kernel node
  void* kernelArgs1[] = {&B_d};
  if (useConstVar) {
    kernelNodeParams.func =
                     reinterpret_cast<void *>(MemcpyToConstSymbolExecKernel);
  } else {
    kernelNodeParams.func = reinterpret_cast<void *>(MemcpyToSymbolExecKernel);
  }
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memcpytosymbolkernel, graph,
                                  dependencies.data(), dependencies.size(),
                                  &kernelNodeParams));
  dependencies.clear();
  dependencies.push_back(memcpytosymbolkernel);

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_B, graph, dependencies.data(),
                                    dependencies.size(), B_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Update the node with source pointer from C_d to A_d
  if (useConstVar) {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                            memcpyToSymbolNode,
                                            HIP_SYMBOL(globalConst), A_d,
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  } else {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(graphExec,
                                            memcpyToSymbolNode,
                                            HIP_SYMBOL(globalIn), A_d,
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  }

  HIP_CHECK(hipGraphLaunch(graphExec, 0));

  // Validating the result
  for (int i = 0; i < SIZE; i++) {
    if (B_h[i] != A_h[i]) {
       WARN("Validation failed B_h[i] " << B_h[i] << "A_h[i] " << A_h[i]);
       REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphExecMemcpyNodeSetParamsToSymbol Functional scenario.
1) Allocate global symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
2) Allocate const symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") {
  SECTION("Check and update with Global Device Symbol Memory") {
    hipGraphExecMemcpyNodeSetParamsToSymbol_GlobalMem(false);
  }
  SECTION("Check and update with Constant Global Device Symbol Memory") {
    hipGraphExecMemcpyNodeSetParamsToSymbol_GlobalMem(true);
  }
}

