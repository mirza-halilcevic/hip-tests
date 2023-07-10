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

#include <vector>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

#include "texture_reference.hh"
#include "utils.hh"
#include "vec4.hh"

namespace cg = cooperative_groups;

template <typename TexelType>
__global__ void tex2DKernel(TexelType* const out, size_t N_x, size_t N_y,
                            hipTextureObject_t tex_obj, size_t width, size_t height,
                            size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  float x = static_cast<float>(tid_x) - N_x / 2;
  x = normalized_coords ? x / width : x;

  float y = static_cast<float>(tid_y) - N_y / 2;
  y = normalized_coords ? y / height : y;

  out[tid_y * width + tid_x] = tex2D<TexelType>(tex_obj, x, y);
}

TEST_CASE("Unit_tex2D_Positive") {
  using TestType = float;

  const auto width = 5;
  const auto height = 3;
  const auto num_iters_x = 3 * width * 2 + 1;
  const auto num_iters_y = 3 * height * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * height * sizeof(vec4<TestType>));
  for (auto i = 0u; i < width * height; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), make_hipExtent(width, height, 0), 0);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModePoint);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(false);
  tex_desc.normalizedCoords = normalized_coords;

  auto address_mode = hipAddressModeClamp;
  tex_desc.addressMode[0] = address_mode;
  tex_desc.addressMode[1] = address_mode;

  ArrayAllocGuard<vec4<TestType>> tex_alloc_d(
      make_hipExtent(tex_h.extent().width, tex_h.extent().height, 0));

  const size_t spitch = tex_h.extent().width * sizeof(vec4<TestType>);
  HIP_CHECK(hipMemcpy2DToArray(tex_alloc_d.ptr(), 0, 0, tex_h.ptr(0), spitch, spitch,
                               tex_h.extent().height, hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = tex_alloc_d.ptr();

  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc,
                                               num_iters_x * num_iters_y * sizeof(vec4<TestType>));

  TextureGuard tex(&res_desc, &tex_desc);

  const dim3 dim_grid{1, 1};
  const dim3 dim_block{num_iters_x, num_iters_y};

  tex2DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      out_alloc_d.ptr(), num_iters_x, num_iters_y, tex.object(), tex_h.extent().width,
      tex_h.extent().height, 0, tex_desc.normalizedCoords);

  std::vector<vec4<TestType>> out_alloc_h(num_iters_x * num_iters_y);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(),
                      num_iters_x * num_iters_y * sizeof(vec4<TestType>), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto j = 0u; j < num_iters_y; ++j) {
    for (auto i = 0u; i < num_iters_x; ++i) {
      INFO("i: " << i);
      INFO("j: " << j);
      INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
      INFO("Address mode: " << AddressModeToString(address_mode));

      float x = static_cast<float>(i) - num_iters_x / 2;
      x = tex_desc.normalizedCoords ? x / tex_h.extent().width : x;
      INFO("x: " << std::fixed << std::setprecision(15) << x);

      float y = static_cast<float>(j) - num_iters_y / 2;
      y = tex_desc.normalizedCoords ? y / tex_h.extent().height : y;
      INFO("y: " << std::fixed << std::setprecision(15) << y);

      auto index = j * tex_h.extent().width + i;

      const auto ref_val = tex_h.Tex2D(x, y, tex_desc);
      CHECK(ref_val.x == out_alloc_h[index].x);
      //   REQUIRE(ref_val.y == out_alloc_h[index].y);
      //   REQUIRE(ref_val.z == out_alloc_h[index].z);
      //   REQUIRE(ref_val.w == out_alloc_h[index].w);
    }
  }
}