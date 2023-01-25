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

#include "stream_performance_common.hh"

class MemPoolSetAttributeBenchmark : public Benchmark<MemPoolSetAttributeBenchmark> {
 public:
  void operator()(const hipMemPoolAttr attribute) {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    int value{0};

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attribute, &value));
    }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const hipMemPoolAttr attribute) {
  MemPoolSetAttributeBenchmark benchmark;
  benchmark.AddSectionName(GetMemPoolAttrSectionName(attribute));
  benchmark.Run(attribute);
}

TEST_CASE("Performance_hipMemPoolSetAttribute") {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
                           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  hipMemPoolAttr attribute = GENERATE(hipMemPoolAttrReleaseThreshold,
                                      hipMemPoolReuseFollowEventDependencies,
                                      hipMemPoolReuseAllowOpportunistic,
                                      hipMemPoolReuseAllowInternalDependencies);
  RunBenchmark(attribute);
}
