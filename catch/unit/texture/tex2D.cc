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

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex2D tex2D
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2D` and read mode set to `hipReadModeElementType`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_tex2D_Positive_ReadModeElementType", "", char, unsigned char, short,
                   unsigned short, int, unsigned int, float) {
  TextureTestParams<TestType> params{make_hipExtent(16, 4, 0), 0, 4};
  params.GenerateTextureDesc();

  TextureTestFixture<TestType> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(32, params.batch_samples);
  dim3 dim_grid(num_blocks, num_blocks);
  dim3 dim_block(num_threads, num_threads);

  size_t num_batches_x = (params.TotalSamplesX() + params.batch_samples - 1) / params.batch_samples;
  size_t num_batches_y = (params.TotalSamplesY() + params.batch_samples - 1) / params.batch_samples;

  int64_t offset_y = -static_cast<int64_t>(params.TotalSamplesY()) / 2;
  int64_t offset_x = -static_cast<int64_t>(params.TotalSamplesX()) / 2;
  for (auto batch = 0u; batch < num_batches_x * num_batches_y; ++batch) {
    const auto batch_x = batch % num_batches_x;
    const auto batch_y = batch / num_batches_x;

    offset_x = (batch_x == 0) ? -static_cast<int64_t>(params.TotalSamplesX()) / 2
                              : offset_x + batch_x * params.batch_samples;
    offset_y = (batch_x == 0) ? offset_y + batch_y * params.batch_samples : offset_y;

    const size_t N_x =
        (batch_x == num_batches_x - 1) && (params.TotalSamplesX() % params.batch_samples)
        ? params.TotalSamplesX() % params.batch_samples
        : params.batch_samples;

    const size_t N_y =
        (batch_y == num_batches_y - 1) && (params.TotalSamplesY() % params.batch_samples)
        ? params.TotalSamplesY() % params.batch_samples
        : params.batch_samples;

    tex2DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), offset_x, offset_y, N_x, N_y, fixture.tex.object(),
        params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto i = 0u; i < N_x * N_y; ++i) {
      float x = i % N_x;
      float y = i / N_x;

      x = GetCoordinate(x, offset_x, params.Width(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      y = GetCoordinate(y, offset_y, params.Height(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);

      INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
      INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
      INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
      INFO("x: " << std::fixed << std::setprecision(16) << x);
      INFO("y: " << std::fixed << std::setprecision(16) << y);

      const auto ref_val = fixture.tex_h.Tex2D(x, y, params.tex_desc);
      REQUIRE(ref_val.x == fixture.out_alloc_h[i].x);
      REQUIRE(ref_val.y == fixture.out_alloc_h[i].y);
      REQUIRE(ref_val.z == fixture.out_alloc_h[i].z);
      REQUIRE(ref_val.w == fixture.out_alloc_h[i].w);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2D` and read mode set to `hipReadModeNormalizedFloat`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_tex2D_Positive_ReadModeNormalizedFloat", "", char, unsigned char, short,
                   unsigned short) {
  TextureTestParams<TestType> params{make_hipExtent(16, 4, 0), 0, 4};
  params.GenerateTextureDesc(hipReadModeNormalizedFloat);

  TextureTestFixture<TestType, true> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(32, params.batch_samples);
  dim3 dim_grid(num_blocks, num_blocks);
  dim3 dim_block(num_threads, num_threads);

  size_t num_batches_x = (params.TotalSamplesX() + params.batch_samples - 1) / params.batch_samples;
  size_t num_batches_y = (params.TotalSamplesY() + params.batch_samples - 1) / params.batch_samples;

  int64_t offset_y = -static_cast<int64_t>(params.TotalSamplesY()) / 2;
  int64_t offset_x = -static_cast<int64_t>(params.TotalSamplesX()) / 2;
  for (auto batch = 0u; batch < num_batches_x * num_batches_y; ++batch) {
    const auto batch_x = batch % num_batches_x;
    const auto batch_y = batch / num_batches_x;

    offset_x = (batch_x == 0) ? -static_cast<int64_t>(params.TotalSamplesX()) / 2
                              : offset_x + batch_x * params.batch_samples;
    offset_y = (batch_x == 0) ? offset_y + batch_y * params.batch_samples : offset_y;

    const size_t N_x =
        (batch_x == num_batches_x - 1) && (params.TotalSamplesX() % params.batch_samples)
        ? params.TotalSamplesX() % params.batch_samples
        : params.batch_samples;

    const size_t N_y =
        (batch_y == num_batches_y - 1) && (params.TotalSamplesY() % params.batch_samples)
        ? params.TotalSamplesY() % params.batch_samples
        : params.batch_samples;

    tex2DKernel<vec4<float>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), offset_x, offset_y, N_x, N_y, fixture.tex.object(),
        params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto i = 0u; i < N_x * N_y; ++i) {
      float x = i % N_x;
      float y = i / N_x;

      x = GetCoordinate(x, offset_x, params.Width(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      y = GetCoordinate(y, offset_y, params.Height(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);

      INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
      INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
      INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
      INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
      INFO("x: " << std::fixed << std::setprecision(16) << x);
      INFO("y: " << std::fixed << std::setprecision(16) << y);

      auto ref_val =
          Vec4Map<TestType>(fixture.tex_h.Tex2D(x, y, params.tex_desc), NormalizeInteger<TestType>);
      REQUIRE(ref_val.x == fixture.out_alloc_h[i].x);
      REQUIRE(ref_val.y == fixture.out_alloc_h[i].y);
      REQUIRE(ref_val.z == fixture.out_alloc_h[i].z);
      REQUIRE(ref_val.w == fixture.out_alloc_h[i].w);
    }
  }
}
