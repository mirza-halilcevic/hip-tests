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
__global__ void tex1DKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj,
                            size_t width, size_t num_subdivisions, bool normalized_coord) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;


  float x = (static_cast<float>(tid) - N / 2) / num_subdivisions;
  x = normalized_coord ? x / width : x;
  out[tid] = tex1D<TexelType>(tex_obj, x);
}

TEST_CASE("Unit_tex1D_Positive") {
  using TestType = float;

  const auto width = 1024;
  const auto num_subdivisions = 512;
  const auto num_iters = 3 * width * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * sizeof(vec4<TestType>));
  for (auto i = 0; i < width; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), width);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModePoint, hipFilterModeLinear);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(false, true);
  tex_desc.normalizedCoords = normalized_coords;

  auto address_mode = hipAddressModeClamp;
  if (normalized_coords) {
    address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                            hipAddressModeMirror);
  } else {
    address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  }
  tex_desc.addressMode[0] = address_mode;

  ArrayAllocGuard<vec4<TestType>> tex_alloc_d(make_hipExtent(tex_h.width(), 0, 0));
  const size_t spitch = tex_h.width() * sizeof(vec4<TestType>);
  HIP_CHECK(hipMemcpy2DToArray(tex_alloc_d.ptr(), 0, 0, tex_h.ptr(), spitch, spitch, 1,
                               hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = tex_alloc_d.ptr();

  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc,
                                               num_iters * sizeof(vec4<TestType>));

  TextureGuard tex(&res_desc, &tex_desc);
  const auto num_threads = std::min<size_t>(1024, num_iters);
  const auto num_blocks = (num_iters + num_threads - 1) / num_threads;
  tex1DKernel<vec4<TestType>>
      <<<num_blocks, num_threads>>>(out_alloc_d.ptr(), num_iters, tex.object(), tex_h.width(),
                                    num_subdivisions, tex_desc.normalizedCoords);

  std::vector<vec4<TestType>> out_alloc_h(num_iters);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), num_iters * sizeof(vec4<TestType>),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto i = 0u; i < out_alloc_h.size(); ++i) {
    INFO("Index: " << i);
    INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
    INFO("Address mode: " << AddressModeToString(address_mode));
    float x = (static_cast<float>(i) - num_iters / 2) / num_subdivisions;
    x = tex_desc.normalizedCoords ? x / tex_h.width() : x;
    INFO("Coordinate: " << std::fixed << std::setprecision(15) << x);
    const auto ref_val = tex_h.Tex1D(x, tex_desc);
    REQUIRE(ref_val.x == out_alloc_h[i].x);
    REQUIRE(ref_val.y == out_alloc_h[i].y);
    REQUIRE(ref_val.z == out_alloc_h[i].z);
    REQUIRE(ref_val.w == out_alloc_h[i].w);
  }
}

template <typename TexelType>
__global__ void tex1DRefKernel(TexelType* const out, size_t N, TexelType* const tex, size_t width,
                               hipTextureDesc tex_desc, size_t num_subdivisions) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;

  TextureReference<TexelType> tex_ref(tex, width);

  float x = (static_cast<float>(tid) - N / 2) / num_subdivisions;
  x = tex_desc.normalizedCoords ? x / width : x;
  out[tid] = tex_ref.Tex1D(x, tex_desc);
}

TEST_CASE("Bla") {
  using TestType = float;

  const auto width = 1024;
  const auto num_subdivisions = 512;
  const auto num_iters = 3 * width * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * sizeof(vec4<TestType>));
  for (auto i = 0; i < width; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModePoint, hipFilterModeLinear);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(false, true);
  tex_desc.normalizedCoords = normalized_coords;

  auto address_mode = hipAddressModeClamp;
  if (normalized_coords) {
    address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                            hipAddressModeMirror);
  } else {
    address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  }
  tex_desc.addressMode[0] = address_mode;

  LinearAllocGuard<vec4<TestType>> out_alloc_h(LinearAllocs::hipHostMalloc,
                                               num_iters * sizeof(vec4<TestType>));
  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc,
                                               num_iters * sizeof(vec4<TestType>));
  const auto num_threads = std::min<size_t>(1024, num_iters);
  const auto num_blocks = (num_iters + num_threads - 1) / num_threads;
  {
    ArrayAllocGuard<vec4<TestType>> tex_alloc_d(make_hipExtent(width, 0, 0));
    const size_t spitch = width * sizeof(vec4<TestType>);
    HIP_CHECK(hipMemcpy2DToArray(tex_alloc_d.ptr(), 0, 0, host_alloc.ptr(), spitch, spitch, 1,
                                 hipMemcpyHostToDevice));

    hipResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = hipResourceTypeArray;
    res_desc.res.array.array = tex_alloc_d.ptr();

    TextureGuard tex(&res_desc, &tex_desc);
    tex1DKernel<vec4<TestType>><<<num_blocks, num_threads>>>(out_alloc_d.ptr(), num_iters,
                                                             tex.object(), width, num_subdivisions,
                                                             tex_desc.normalizedCoords);

    HIP_CHECK(hipMemcpy(out_alloc_h.ptr(), out_alloc_d.ptr(), num_iters * sizeof(vec4<TestType>),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
  }

  LinearAllocGuard<vec4<TestType>> tex_alloc_d(LinearAllocs::hipMalloc,
                                               width * sizeof(vec4<TestType>));
  HIP_CHECK(hipMemcpy(tex_alloc_d.ptr(), host_alloc.ptr(), width * sizeof(vec4<TestType>),
                      hipMemcpyHostToDevice));

  tex1DRefKernel<vec4<TestType>><<<num_blocks, num_threads>>>(
      out_alloc_d.ptr(), num_iters, tex_alloc_d.ptr(), width, tex_desc, num_subdivisions);

  LinearAllocGuard<vec4<TestType>> ref_alloc_h(LinearAllocs::hipHostMalloc,
                                               num_iters * sizeof(vec4<TestType>));
  HIP_CHECK(hipMemcpy(ref_alloc_h.ptr(), out_alloc_d.ptr(), num_iters * sizeof(vec4<TestType>),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto i = 0u; i < num_iters; ++i) {
    INFO("Index: " << i);
    INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
    INFO("Address mode: " << AddressModeToString(address_mode));
    float x = (static_cast<float>(i) - num_iters / 2) / num_subdivisions;
    x = tex_desc.normalizedCoords ? x / width : x;
    INFO("Coordinate: " << std::fixed << std::setprecision(15) << x);
    REQUIRE(ref_alloc_h.ptr()[i].x == out_alloc_h.ptr()[i].x);
    REQUIRE(ref_alloc_h.ptr()[i].y == out_alloc_h.ptr()[i].y);
    REQUIRE(ref_alloc_h.ptr()[i].z == out_alloc_h.ptr()[i].z);
    REQUIRE(ref_alloc_h.ptr()[i].w == out_alloc_h.ptr()[i].w);
  }
}