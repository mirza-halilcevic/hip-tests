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
__global__ void tex1DLayeredKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj,
                                   int layers, size_t width, size_t num_subdivisions,
                                   bool normalized_coord) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;


  float x = (static_cast<float>(tid) - N / 2) / num_subdivisions;
  x = normalized_coord ? x / width : x;
  for (int i = 0; i < layers; ++i) {
    out[tid + i * N] = tex1DLayered<TexelType>(tex_obj, x, i);
  }
}

TEST_CASE("Unit_tex1DLayered_Positive") {
  using TestType = float;

  const auto layers = 2;
  const auto width = 512;
  const auto num_subdivisions = 512;
  const auto out_of_bound_mult = 3;
  // This defines the number of kernel threads that will perform texture sampling.
  // The number of integer indices for a texture is equal to the width
  // To perform non-integer indexing, each interval between two adjacent integer indices is divided
  // into num_subdivisions sub-intervals(* num_subdivision)
  // To perform out of bounds indexing, out_of_bound_mult defines how many times larger than the
  // largest valid index the largest out of bound index should be(*out_of_bound_mult)
  // To perform indexing with negative indices, an interval symmetrical about zero is formed(*2 + 1)
  const auto num_iters = out_of_bound_mult * width * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              layers * width * sizeof(vec4<TestType>));
  for (auto i = 0; i < layers * width; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), width, layers);

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

  // Da bi bio 1D layered array, height mora biti 0
  ArrayAllocGuard<vec4<TestType>> tex_alloc_d(make_hipExtent(tex_h.width(), 0, layers),
                                              hipArrayLayered);
  hipMemcpy3DParms memcpy_params = {0};
  memcpy_params.dstArray = tex_alloc_d.ptr();
  // Premda kod kreiranja array-a height mora biti nula, u naredne dvije linije mora biti 1
  memcpy_params.extent = make_hipExtent(width, 1, layers);
  memcpy_params.srcPtr = make_hipPitchedPtr(tex_h.ptr(0), width * sizeof(vec4<TestType>), width, 1);
  memcpy_params.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&memcpy_params));


  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = tex_alloc_d.ptr();

  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc,
                                               layers * num_iters * sizeof(vec4<TestType>));

  TextureGuard tex(&res_desc, &tex_desc);
  const auto num_threads = std::min<size_t>(1024, num_iters);
  const auto num_blocks = (num_iters + num_threads - 1) / num_threads;
  tex1DLayeredKernel<vec4<TestType>>
      <<<num_blocks, num_threads>>>(out_alloc_d.ptr(), num_iters, tex.object(), layers,
                                    tex_h.width(), num_subdivisions, tex_desc.normalizedCoords);

  std::vector<vec4<TestType>> out_alloc_h(layers * num_iters);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(),
                      layers * num_iters * sizeof(vec4<TestType>), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto i = 0u; i < out_alloc_h.size(); ++i) {
    int layer = i / num_iters;
    float x = i % num_iters;
    x = (x - num_iters / 2) / num_subdivisions;
    x = tex_desc.normalizedCoords ? x / tex_h.width() : x;

    INFO("Filter mode: " << FilteringModeToString(filter_mode));
    INFO("Address mode: " << AddressModeToString(address_mode));
    INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
    INFO("Layer: " << layer);
    INFO("Coordinate: " << std::fixed << std::setprecision(15) << x);

    const auto ref_val = tex_h.Tex1DLayered(x, layer, tex_desc);
    REQUIRE(ref_val.x == out_alloc_h[i].x);
    REQUIRE(ref_val.y == out_alloc_h[i].y);
    REQUIRE(ref_val.z == out_alloc_h[i].z);
    REQUIRE(ref_val.w == out_alloc_h[i].w);
  }
}
