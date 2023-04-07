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
#include <resource_guards.hh>

#include "graph_engine.hh"
#include "macros.hh"

#define INPUT_TYPE double
#define OUTPUT_TYPE double

INPUT_GENERATOR_WRAPPER_DEF(__sin) {
  uint32_t rand = rocrand(states);
  return *reinterpret_cast<float*>(&rand);
}

TEST_VALUE_GENERATOR_WRAPPER_DEF(__sin) { return sin(x); }

REFERENCE_GENERATOR_WRAPPER_DEF(__sin) {
  return static_cast<long double (*)(long double)>(std::sin)(x);
}

VALIDATOR_WRAPPER_DEF(__sin) { return ULPValidatorBuilderFactory<double>(2)(y2).match(y1); }

INPUT_GENERATOR_DEF(__sin);
TEST_VALUE_GENERATOR_DEF(__sin);
REFERENCE_GENERATOR_DEF(__sin);
VALIDATOR_DEF(__sin);

TEST_CASE("Unit___sin_Positive_Random") {
  MATH_TEST_RANDOM(__sin, 100'000'000, std::numeric_limits<uint32_t>::max());
}