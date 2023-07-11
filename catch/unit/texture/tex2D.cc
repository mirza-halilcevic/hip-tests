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

  float x = (static_cast<float>(tid_x) - N_x / 2) / num_subdivisions;
  x = normalized_coords ? x / width : x;

  float y = (static_cast<float>(tid_y) - N_y / 2) / num_subdivisions;
  y = normalized_coords ? y / height : y;

  auto res = tex2D<TexelType>(tex_obj, x, y);
  auto index = tid_y * N_x + tid_x;

  // printf("tid_x: %d tid_y: %d x: %f y: %f res: %f index: %d\n", tid_x, tid_y, x, y, res.x,
  // index);

  out[index] = res;
}

TEST_CASE("Unit_tex2D_Positive") {
  using TestType = float;

  const auto width = 16;
  const auto height = 4;
  const auto num_subdivisions = 5;
  const auto num_iters_x = 3 * width * num_subdivisions * 2 + 1;
  const auto num_iters_y = 3 * height * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * height * sizeof(vec4<TestType>));
  for (auto i = 0u; i < width * height; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 1);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), make_hipExtent(width, height, 0), 0);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModeLinear);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(false);
  tex_desc.normalizedCoords = normalized_coords;

  auto address_mode_x = hipAddressModeMirror, address_mode_y = hipAddressModeMirror;
  // if (normalized_coords) {
  //   address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
  //                           hipAddressModeMirror);
  //   address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
  //                           hipAddressModeMirror);
  // } else {
  //   address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  //   address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  // }
  tex_desc.addressMode[0] = address_mode_x;
  tex_desc.addressMode[1] = address_mode_y;

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

  const auto num_threads_x = std::min<size_t>(32, num_iters_x);
  const auto num_blocks_x = (num_iters_x + num_threads_x - 1) / num_threads_x;

  const auto num_threads_y = std::min<size_t>(32, num_iters_y);
  const auto num_blocks_y = (num_iters_y + num_threads_y - 1) / num_threads_y;

  const dim3 dim_grid{num_blocks_x, num_blocks_y};
  const dim3 dim_block{num_threads_x, num_threads_y};

  tex2DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      out_alloc_d.ptr(), num_iters_x, num_iters_y, tex.object(), tex_h.extent().width,
      tex_h.extent().height, num_subdivisions, tex_desc.normalizedCoords);
  HIP_CHECK(hipGetLastError());

  std::vector<vec4<TestType>> out_alloc_h(num_iters_x * num_iters_y);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(),
                      num_iters_x * num_iters_y * sizeof(vec4<TestType>), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // for (auto v : out_alloc_h) {
  //   std::cout << v.x << std::endl;
  // }

  for (auto j = 0u; j < num_iters_y; ++j) {
    for (auto i = 0u; i < num_iters_x; ++i) {
      INFO("i: " << i);
      INFO("j: " << j);
      INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
      INFO("Address mode X: " << AddressModeToString(address_mode_x));
      INFO("Address mode Y: " << AddressModeToString(address_mode_y));

      float x = (static_cast<float>(i) - num_iters_x / 2) / num_subdivisions;
      x = tex_desc.normalizedCoords ? x / tex_h.extent().width : x;
      INFO("x: " << std::fixed << std::setprecision(15) << x);

      float y = (static_cast<float>(j) - num_iters_y / 2) / num_subdivisions;
      y = tex_desc.normalizedCoords ? y / tex_h.extent().height : y;
      INFO("y: " << std::fixed << std::setprecision(15) << y);

      auto index = j * num_iters_x + i;

      const auto ref_val = tex_h.Tex2D(x, y, tex_desc);
      CHECK(ref_val.x == out_alloc_h[index].x);
      //   REQUIRE(ref_val.y == out_alloc_h[index].y);
      //   REQUIRE(ref_val.z == out_alloc_h[index].z);
      //   REQUIRE(ref_val.w == out_alloc_h[index].w);
    }
  }
}

__global__ void kernel2D(hipTextureObject_t tex_obj) {
  const auto v = tex2D<float2>(tex_obj, 0, 0);
  printf("x:%1.10f, y:%1.10f\n", v.x, v.y);
}

TEST_CASE("BLA") {
  const int width = 2;
  const int height = 2;

  std::vector<float2> vec(width * height);
  for (auto i = 0u; i < vec.size(); ++i) {
    vec[i].x = i + 1;
    vec[i].y = i + 1;
  }

  hipArray_t array;
  const auto desc = hipCreateChannelDesc<float2>();
  HIP_CHECK(hipMalloc3DArray(&array, &desc, make_hipExtent(width, height, 0), 0));
  const auto spitch = width * sizeof(float2);
  HIP_CHECK(
      hipMemcpy2DToArray(array, 0, 0, vec.data(), spitch, spitch, height, hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = array;

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.normalizedCoords = false;
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.addressMode[0] = hipAddressModeClamp;
  tex_desc.addressMode[1] = hipAddressModeClamp;
  tex_desc.readMode = hipReadModeElementType;

  hipTextureObject_t tex;
  HIP_CHECK(hipCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));

  kernel2D<<<1, 1>>>(tex);
  HIP_CHECK(hipDeviceSynchronize());
}