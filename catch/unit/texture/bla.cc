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

template <typename Vec> __global__ void kernel(hipTextureObject_t tex_obj) {
  const auto v = tex1D<Vec>(tex_obj, 1025);
  printf("%u\n", v.x);
  printf("%u\n", v.y);
  printf("%u\n", v.z);
  printf("%u\n", v.w);
}

TEST_CASE("Bla") {
  vec4<char> vec;
  SetVec4<char>(vec, 0);
  const int height = 1;
  const int width = 1024;

  using T = vec4<unsigned int>;

  std::vector<T> h_data(width * height);
  for (auto i = 0u; i < h_data.size(); ++i) {
    SetVec4<unsigned int>(h_data[i], i);
  }

  hipChannelFormatDesc channel_desc = hipCreateChannelDesc<T>();

  ArrayAllocGuard<T> arr_alloc(make_hipExtent(1024, 0, 0));

  hipArray_t hip_arr;
  HIP_CHECK(hipMallocArray(&hip_arr, &channel_desc, width, height));

  const size_t spitch = width * sizeof(T);
  HIP_CHECK(hipMemcpy2DToArray(arr_alloc.ptr(), 0, 0, h_data.data(), spitch, spitch, height,
                               hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = arr_alloc.ptr();

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = hipAddressModeWrap;
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.normalizedCoords = false;

  hipTextureObject_t tex_obj = 0;
  HIP_CHECK(hipCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

  kernel<T><<<1, 1>>>(tex_obj);
  HIP_CHECK(hipDeviceSynchronize());
}