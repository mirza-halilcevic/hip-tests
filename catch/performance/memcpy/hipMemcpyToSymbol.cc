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

__device__ int devSymbol[1_MB];

class MemcpyToSymbolBenchmark : public Benchmark<MemcpyToSymbolBenchmark> {
 public:
  void operator()(const void* source, size_t size=1, size_t offset=0) {
    TIMED_SECTION(TIMER_TYPE_EVENT) {
      HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), source, size, offset));
    }
  }
};

static void RunBenchmark(const void* source, size_t size=1, size_t offset=0) {
  MemcpyToSymbolBenchmark benchmark;
  benchmark.Configure(100, 1000);
  auto time = benchmark.Run(source, size, offset, true);
  std::cout << time << " ms" << std::endl;
}

TEST_CASE("Performance_hipMemcpyToSymbol_SingularValue") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  int set{42};
  RunBenchmark(&set);
}

TEST_CASE("Performance_hipMemcpyToSymbol_ArrayValue") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  size_t size = GENERATE(1_KB, 4_KB, 1_MB);
  int array[size];
  std::fill_n(array, size, 42);

  RunBenchmark(array, sizeof(int) * size);
}

TEST_CASE("Performance_hipMemcpyToSymbol_WithOffset") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  size_t size = GENERATE(1_KB, 4_KB, 1_MB);
  int array[size];
  std::fill_n(array, size, 42);

  size_t offset = GENERATE_REF(0, size / 2);
  RunBenchmark(array + offset, sizeof(int) * (size - offset), offset * sizeof(int));
}
