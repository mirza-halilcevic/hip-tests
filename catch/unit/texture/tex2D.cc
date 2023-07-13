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

TEST_CASE("Unit_tex2D_Positive") {
  using TestType = float;

  TextureTestParams<TestType> params = {0};
  params.extent = make_hipExtent(16, 4, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc();

  TextureTestFixture<TestType> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  tex2DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), fixture.tex.object(),
      params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords);
  HIP_CHECK(hipGetLastError());

  fixture.LoadOutput();

  for (auto j = 0u; j < params.NumItersY(); ++j) {
    for (auto i = 0u; i < params.NumItersX(); ++i) {
      float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                              params.tex_desc.normalizedCoords);
      float y = GetCoordinate(j, params.NumItersY(), params.Height(), params.num_subdivisions,
                              params.tex_desc.normalizedCoords);

      INFO("i: " << i);
      INFO("j: " << j);
      INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
      INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
      INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
      INFO("x: " << std::fixed << std::setprecision(16) << x);
      INFO("y: " << std::fixed << std::setprecision(16) << y);

      auto index = j * params.NumItersX() + i;

      const auto ref_val = fixture.tex_h.Tex2D(x, y, params.tex_desc);
      REQUIRE(ref_val.x == fixture.out_alloc_h[index].x);
      REQUIRE(ref_val.y == fixture.out_alloc_h[index].y);
      REQUIRE(ref_val.z == fixture.out_alloc_h[index].z);
      REQUIRE(ref_val.w == fixture.out_alloc_h[index].w);
    }
  }
}