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
__global__ void tex3DKernel(TexelType* const out, size_t N_x, size_t N_y, size_t N_z,
                            hipTextureObject_t tex_obj, size_t width, size_t height, size_t depth,
                            size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  const auto tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (tid_z >= N_z) return;

  float x = (static_cast<float>(tid_x) - N_x / 2) / num_subdivisions;
  x = normalized_coords ? x / width : x;

  float y = (static_cast<float>(tid_y) - N_y / 2) / num_subdivisions;
  y = normalized_coords ? y / height : y;

  float z = (static_cast<float>(tid_z) - N_z / 2) / num_subdivisions;
  z = normalized_coords ? z / depth : z;

  out[tid_z * N_x * N_y + tid_y * N_x + tid_x] = tex3D<TexelType>(tex_obj, x, y, z);
}

static auto GenerateAddressModes(bool normalized_coords) {
  auto address_mode_x = hipAddressModeWrap;
  auto address_mode_y = address_mode_x;
  auto address_mode_z = address_mode_y;
  //   if (normalized_coords) {
  //     address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
  //                               hipAddressModeMirror);
  //     address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
  //                               hipAddressModeMirror);
  //     address_mode_z = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
  //                               hipAddressModeMirror);
  //   } else {
  //     address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  //     address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  //     address_mode_z = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  //   }
  return std::make_tuple(address_mode_x, address_mode_y, address_mode_z);
}

TEST_CASE("Unit_tex3D_Positive") {
  using TestType = float;

  const auto width = 2;
  const auto height = 2;
  const auto depth = 2;
  const auto num_subdivisions = 4;
  const auto num_iters_x = 3 * width * num_subdivisions * 2 + 1;
  const auto num_iters_y = 3 * height * num_subdivisions * 2 + 1;
  const auto num_iters_z = 3 * depth * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * height * depth * sizeof(vec4<TestType>));
  for (auto i = 0u; i < width * height * depth; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), make_hipExtent(width, height, depth), 0);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModeLinear);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(true);
  tex_desc.normalizedCoords = normalized_coords;

  const auto [address_mode_x, address_mode_y, address_mode_z] =
      GenerateAddressModes(normalized_coords);
  tex_desc.addressMode[0] = address_mode_x;
  tex_desc.addressMode[1] = address_mode_y;
  tex_desc.addressMode[2] = address_mode_z;

  ArrayAllocGuard<vec4<TestType>> tex_alloc_d(
      make_hipExtent(tex_h.extent().width, tex_h.extent().height, tex_h.extent().depth));

  hipMemcpy3DParms memcpy_params = {0};
  memcpy_params.dstArray = tex_alloc_d.ptr();
  memcpy_params.extent = make_hipExtent(width, height, depth);
  memcpy_params.srcPtr =
      make_hipPitchedPtr(tex_h.ptr(0), width * sizeof(vec4<TestType>), width, height);
  memcpy_params.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&memcpy_params));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = tex_alloc_d.ptr();

  LinearAllocGuard<vec4<TestType>> out_alloc_d(
      LinearAllocs::hipMalloc, num_iters_x * num_iters_y * num_iters_z * sizeof(vec4<TestType>));

  TextureGuard tex(&res_desc, &tex_desc);

  const auto num_threads_x = std::min<size_t>(10, num_iters_x);
  const auto num_blocks_x = (num_iters_x + num_threads_x - 1) / num_threads_x;

  const auto num_threads_y = std::min<size_t>(10, num_iters_y);
  const auto num_blocks_y = (num_iters_y + num_threads_y - 1) / num_threads_y;

  const auto num_threads_z = std::min<size_t>(10, num_iters_z);
  const auto num_blocks_z = (num_iters_z + num_threads_z - 1) / num_threads_z;

  const dim3 dim_grid{num_blocks_x, num_blocks_y, num_blocks_z};
  const dim3 dim_block{num_threads_x, num_threads_y, num_threads_z};

  tex3DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      out_alloc_d.ptr(), num_iters_x, num_iters_y, num_iters_z, tex.object(), tex_h.extent().width,
      tex_h.extent().height, tex_h.extent().depth, num_subdivisions, tex_desc.normalizedCoords);
  HIP_CHECK(hipGetLastError());

  std::vector<vec4<TestType>> out_alloc_h(num_iters_x * num_iters_y * num_iters_z);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(),
                      num_iters_x * num_iters_y * num_iters_z * sizeof(vec4<TestType>),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto k = 0u; k < num_iters_z; ++k) {
    for (auto j = 0u; j < num_iters_y; ++j) {
      for (auto i = 0u; i < num_iters_x; ++i) {
        INFO("i: " << i);
        INFO("j: " << j);
        INFO("k: " << k);
        INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
        INFO("Address mode X: " << AddressModeToString(address_mode_x));
        INFO("Address mode Y: " << AddressModeToString(address_mode_y));
        INFO("Address mode Z: " << AddressModeToString(address_mode_z));

        float x = (static_cast<float>(i) - num_iters_x / 2) / num_subdivisions;
        x = tex_desc.normalizedCoords ? x / tex_h.extent().width : x;
        INFO("x: " << std::fixed << std::setprecision(30) << x);

        float y = (static_cast<float>(j) - num_iters_y / 2) / num_subdivisions;
        y = tex_desc.normalizedCoords ? y / tex_h.extent().height : y;
        INFO("y: " << std::fixed << std::setprecision(30) << y);

        float z = (static_cast<float>(k) - num_iters_z / 2) / num_subdivisions;
        z = tex_desc.normalizedCoords ? z / tex_h.extent().depth : z;
        INFO("z: " << std::fixed << std::setprecision(30) << z);

        auto index = k * num_iters_x * num_iters_y + j * num_iters_x + i;

        const auto ref_val = tex_h.Tex3D(x, y, z, tex_desc);
        REQUIRE(ref_val.x == out_alloc_h[index].x);
        REQUIRE(ref_val.y == out_alloc_h[index].y);
        REQUIRE(ref_val.z == out_alloc_h[index].z);
        REQUIRE(ref_val.w == out_alloc_h[index].w);
      }
    }
  }
}