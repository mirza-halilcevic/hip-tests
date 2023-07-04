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
__global__ void tex1DfetchKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;

  out[tid] = tex1Dfetch<TexelType>(tex_obj, tid);
}

template <typename T, typename RT>
std::vector<RT> Tex1DFetchTest(TextureReference<T>& tex_ref, hipTextureDesc& tex_desc,
                               const size_t num_iters) {
  LinearAllocGuard<T> tex_alloc_d(LinearAllocs::hipMalloc, tex_ref.width() * sizeof(T));
  HIP_CHECK(hipMemcpy(tex_alloc_d.ptr(), tex_ref.ptr(), tex_ref.width() * sizeof(T),
                      hipMemcpyHostToDevice));

  LinearAllocGuard<RT> out_alloc_d(LinearAllocs::hipMalloc, num_iters * sizeof(RT));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<T>();
  res_desc.res.linear.sizeInBytes = tex_ref.width() * sizeof(T);

  TextureGuard tex(&res_desc, &tex_desc);
  const auto num_threads = std::min<size_t>(1024, num_iters);
  const auto num_blocks = (num_iters + num_threads - 1) / num_threads;
  tex1DfetchKernel<RT><<<num_blocks, num_threads>>>(out_alloc_d.ptr(), num_iters, tex.object());

  std::vector<RT> out_alloc_h(num_iters);
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), num_iters * sizeof(RT),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  return out_alloc_h;
}

#ifdef __LP64__
#define LP_ENABLED_TYPES
#elif
#define LP_ENABLED_TYPES , long, unsigned long, long long, unsigned long long
#endif

TEMPLATE_TEST_CASE("Unit_tex1Dfetch_ReadModeElementType_Positive_Basic", "", char, unsigned char,
                   short, unsigned short, int, unsigned int, float LP_ENABLED_TYPES) {
  TextureReference<vec4<TestType>> tex_h(1024);
  // TODO - Need some negative values for signed types.
  tex_h.Fill([](size_t x) { return MakeVec4<TestType>(x); });

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.normalizedCoords = false;

  const auto address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  DYNAMIC_SECTION("Address mode: " << AddressModeToString(address_mode)) {
    tex_desc.addressMode[0] = address_mode;

    const auto res =
        Tex1DFetchTest<vec4<TestType>, vec4<TestType>>(tex_h, tex_desc, tex_h.width() * 2);
    for (auto i = 0u; i < res.size(); ++i) {
      INFO("Index: " << i);
      const auto ref_val = tex_h.Fetch1D(i, tex_desc);
      REQUIRE(ref_val.x == res[i].x);
      REQUIRE(ref_val.y == res[i].y);
      REQUIRE(ref_val.z == res[i].z);
      REQUIRE(ref_val.w == res[i].w);
    }
  }
}


TEMPLATE_TEST_CASE("Unit_tex1Dfetch_ReadModeNormalizedFloat_Positive_Basic", "", char,
                   unsigned char, short, unsigned short) {
  TextureReference<vec4<TestType>> tex_h(1024);
  // TODO - Need some negative values for signed types.
  tex_h.Fill([](size_t x) { return MakeVec4<TestType>(x); });

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeNormalizedFloat;
  tex_desc.normalizedCoords = false;

  const auto address_mode = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  DYNAMIC_SECTION("Address mode: " << AddressModeToString(address_mode)) {
    tex_desc.addressMode[0] = address_mode;

    const auto res =
        Tex1DFetchTest<vec4<TestType>, vec4<float>>(tex_h, tex_desc, tex_h.width() * 2);
    for (auto i = 0u; i < res.size(); ++i) {
      INFO("Index: " << i);
      const auto ref_val = Vec4Map<TestType>(tex_h.Fetch1D(i, tex_desc),
                                             [](TestType x) { return NormalizeInteger(x); });
      REQUIRE(ref_val.x == res[i].x);
      REQUIRE(ref_val.y == res[i].y);
      REQUIRE(ref_val.z == res[i].z);
      REQUIRE(ref_val.w == res[i].w);
    }
  }
}
