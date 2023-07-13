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
__global__ void tex2DKernel(TexelType* const out, int64_t offset_x, int64_t offset_y, size_t N_x,
                            size_t N_y, hipTextureObject_t tex_obj, size_t width, size_t height,
                            size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  float x = (static_cast<float>(tid_x) + offset_x) / num_subdivisions;
  x = normalized_coords ? x / width : x;

  float y = (static_cast<float>(tid_y) + offset_y) / num_subdivisions;
  y = normalized_coords ? y / height : y;

  out[tid_y * N_x + tid_x] = tex2D<TexelType>(tex_obj, x, y);
}

static auto GenerateAddressModes(bool normalized_coords) {
  auto address_mode_x = hipAddressModeClamp;
  auto address_mode_y = address_mode_x;
  if (normalized_coords) {
    address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                              hipAddressModeMirror);
    address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                              hipAddressModeMirror);
  } else {
    address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
    address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
  }
  return std::make_tuple(address_mode_x, address_mode_y);
}

TEST_CASE("Unit_tex2D_Positive") {
  using TestType = float;

  const auto width = 16;
  const auto height = 4;
  const auto num_subdivisions = 4;
  const int64_t total_samples_x = 3 * width * num_subdivisions * 2 + 1;
  const int64_t total_samples_y = 3 * height * num_subdivisions * 2 + 1;

  LinearAllocGuard<vec4<TestType>> host_alloc(LinearAllocs::hipHostMalloc,
                                              width * height * sizeof(vec4<TestType>));
  for (auto i = 0u; i < width * height; ++i) {
    SetVec4<TestType>(host_alloc.ptr()[i], i + 7);
  }

  TextureReference<vec4<TestType>> tex_h(host_alloc.ptr(), make_hipExtent(width, height, 0), 0);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  const auto filter_mode = GENERATE(hipFilterModePoint, hipFilterModeLinear);
  tex_desc.filterMode = filter_mode;

  const bool normalized_coords = GENERATE(false, true);
  tex_desc.normalizedCoords = normalized_coords;

  const auto [address_mode_x, address_mode_y] = GenerateAddressModes(normalized_coords);
  tex_desc.addressMode[0] = address_mode_x;
  tex_desc.addressMode[1] = address_mode_y;

  ArrayAllocGuard<vec4<TestType>> tex_alloc_d(
      make_hipExtent(tex_h.extent().width, tex_h.extent().height, 0));

  const size_t spitch = tex_h.extent().width * sizeof(vec4<TestType>);
  HIP_CHECK(hipMemcpy2DToArray(tex_alloc_d.ptr(), 0, 0, tex_h.ptr(0), spitch, spitch,
                               tex_h.extent().height, hipMemcpyHostToDevice));

  hipDeviceProp_t props = {};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  uint64_t device_memory = props.totalGlobalMem * 0.80 - width * sizeof(vec4<TestType>);

  const int64_t max_samples = device_memory / sizeof(vec4<TestType>);
  const auto total_samples = total_samples_x * total_samples_y;
  const unsigned int batch_samples = std::sqrt(std::min(total_samples, max_samples));

  size_t num_batches_x = (total_samples_x + batch_samples - 1) / batch_samples;
  size_t num_batches_y = (total_samples_y + batch_samples - 1) / batch_samples;

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = tex_alloc_d.ptr();

  size_t out_alloc_d_size = batch_samples * batch_samples * sizeof(vec4<TestType>);
  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc, out_alloc_d_size);
  std::vector<vec4<TestType>> out_alloc_h(batch_samples * batch_samples);

  TextureGuard tex(&res_desc, &tex_desc);

  const auto num_threads_x = std::min<unsigned int>(32, batch_samples);
  const auto num_blocks_x = (batch_samples + num_threads_x - 1) / num_threads_x;

  const auto num_threads_y = std::min<unsigned int>(32, batch_samples);
  const auto num_blocks_y = (batch_samples + num_threads_y - 1) / num_threads_y;

  const dim3 dim_grid{num_blocks_x, num_blocks_y};
  const dim3 dim_block{num_threads_x, num_threads_y};

  int64_t offset_y = -static_cast<int64_t>(total_samples_y) / 2;
  int64_t offset_x = -static_cast<int64_t>(total_samples_x) / 2;
  for (auto batch = 0u; batch < num_batches_x * num_batches_y; ++batch) {
    const auto batch_x = batch % num_batches_x;
    const auto batch_y = batch / num_batches_x;

    offset_x = (batch_x == 0) ? -static_cast<int64_t>(total_samples_x) / 2
                              : offset_x + batch_x * batch_samples;
    offset_y = (batch_x == 0) ? offset_y + batch_y * batch_samples : offset_y;

    const size_t N_x = (batch_x == num_batches_x - 1) && (total_samples_x % batch_samples)
        ? total_samples_x % batch_samples
        : batch_samples;

    const size_t N_y = (batch_y == num_batches_y - 1) && (total_samples_y % batch_samples)
        ? total_samples_y % batch_samples
        : batch_samples;

    tex2DKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
        out_alloc_d.ptr(), offset_x, offset_y, N_x, N_y, tex.object(), tex_h.extent().width,
        tex_h.extent().height, num_subdivisions, tex_desc.normalizedCoords);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), N_x * N_y * sizeof(vec4<TestType>),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for (auto i = 0u; i < N_x * N_y; ++i) {
      float x = i % N_x;
      x = (x + offset_x) / num_subdivisions;
      x = tex_desc.normalizedCoords ? x / tex_h.extent().width : x;

      float y = i / N_x;
      y = (y + offset_y) / num_subdivisions;
      y = tex_desc.normalizedCoords ? y / tex_h.extent().height : y;

      INFO("Filtering  mode: " << FilteringModeToString(filter_mode));
      INFO("Normalized coordinates: " << std::boolalpha << normalized_coords);
      INFO("Address mode X: " << AddressModeToString(address_mode_x));
      INFO("Address mode Y: " << AddressModeToString(address_mode_y));
      INFO("x: " << std::fixed << std::setprecision(30) << x);
      INFO("y: " << std::fixed << std::setprecision(30) << y);

      const auto ref_val = tex_h.Tex2D(x, y, tex_desc);
      REQUIRE(ref_val.x == out_alloc_h[i].x);
      REQUIRE(ref_val.y == out_alloc_h[i].y);
      REQUIRE(ref_val.z == out_alloc_h[i].z);
      REQUIRE(ref_val.w == out_alloc_h[i].w);
    }
  }
}

// __global__ void kernel2D(hipTextureObject_t tex_obj, float x, float y) {
//   const auto v = tex2D<float2>(tex_obj, x, y);

//   printf("x:%1.10f, y:%1.10f\n", v.x, v.y);
// }

// template <size_t fractional_bits> float FloatToNBitFractional(float x) {
//   const auto fixed_point = static_cast<uint16_t>(roundf(x * (1 << fractional_bits)));

//   return static_cast<float>(fixed_point) / static_cast<float>(1 << fractional_bits);
// }

// std::tuple<int, FixedPoint<8>, float> blahem(float x) {
//   const auto xB = x - 0.5f;

//   const auto i = static_cast<int>(floorf(xB));

//   const FixedPoint<8> alpha = xB - i;

//   return {i, alpha, xB};
// }

// TEST_CASE("BLA") {
//   const int width = 2;

//   const int height = 2;


//   std::vector<float2> vec(width * height);

//   for (auto i = 0u; i < vec.size(); ++i) {
//     vec[i].x = i + 1;

//     vec[i].y = i + 1;
//   }


//   hipArray_t array;

//   const auto desc = hipCreateChannelDesc<float2>();

//   HIP_CHECK(hipMalloc3DArray(&array, &desc, make_hipExtent(width, height, 0), 0));

//   const auto spitch = width * sizeof(float2);

//   HIP_CHECK(

//       hipMemcpy2DToArray(array, 0, 0, vec.data(), spitch, spitch, height,
//       hipMemcpyHostToDevice));


//   hipResourceDesc res_desc;

//   memset(&res_desc, 0, sizeof(res_desc));

//   res_desc.resType = hipResourceTypeArray;

//   res_desc.res.array.array = array;


//   hipTextureDesc tex_desc;

//   memset(&tex_desc, 0, sizeof(tex_desc));

//   tex_desc.normalizedCoords = true;

//   tex_desc.filterMode = hipFilterModeLinear;

//   tex_desc.addressMode[0] = hipAddressModeBorder;

//   tex_desc.addressMode[1] = hipAddressModeBorder;

//   tex_desc.readMode = hipReadModeElementType;


//   hipTextureObject_t tex;

//   HIP_CHECK(hipCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));


//   float x_og = 0.500000000000000000000000000000;

//   float y_og = -0.100000001490116119384765625000;

//   float x = x_og * width;

//   float y = y_og * height;

//   const auto bla = [&](int x, int y) {
//     if (0 > x || x > width - 1) return 0.f;

//     if (0 > y || y > height - 1) return 0.f;

//     return vec[y * width + x].x;
//   };


//   auto [i, alpha, xB] = blahem(x);

//   auto [j, beta, xY] = blahem(y);

//   float values[] = {0, 1, 2, 3, 4};

//   // for (auto i = 0; i < 5; ++i) {
//   //   for (auto j = 0; j < 5; ++j) {
//   //     for (auto k = 0; k < 5; ++k) {
//   //       for (auto l = 0; l < 5; ++l) {
//   //         // const auto res = ((1.0f - alpha) * (1.0f - beta)) * values[i] +
//   //         //     (alpha * (1.0f - beta)) * values[j] + ((1.0f - alpha) * beta) * values[k] +
//   //         //     (alpha * beta) * values[l];
//   //         if (res == 0.125f) {
//   //           std::cout << values[i] << " " << values[j] << " " << values[k] << " " << values[l]
//   //                     << std::endl;
//   //         }
//   //       }
//   //     }
//   //   }
//   // }

//   const auto T_i0j0 = bla(i, j);

//   const auto T_i1j0 = bla(i + 1, j);

//   const auto T_i0j1 = bla(i, j + 1);

//   const auto T_i1j1 = bla(i + 1, j + 1);

//   const FixedPoint<8> one{1.0f};

//   const auto res = ((one - alpha) * (one - beta)).GetFloat() * T_i0j0 +
//       (alpha * (one - beta)).GetFloat() * T_i1j0 + ((one - alpha) * beta).GetFloat() * T_i0j1 +
//       (alpha * beta).GetFloat() * T_i1j1;

//   kernel2D<<<1, 1>>>(tex, x_og, y_og);

//   HIP_CHECK(hipDeviceSynchronize());

//   std::cout << i << " " << j << std::endl;

//   std::cout << alpha.GetFloat() << " " << beta.GetFloat() << std::endl;

//   std::cout << T_i0j0 << " " << T_i0j1 << " " << T_i1j0 << " " << T_i1j1 << std::endl;

//   std::cout << std::fixed << std::setprecision(10) << res << std::endl;
// }