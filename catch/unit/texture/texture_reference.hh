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

#include <algorithm>

#include <hip/hip_runtime_api.h>

#include <hip/hip_cooperative_groups.h>

#include <resource_guards.hh>

template <typename TexelType> class TextureReference {
 public:
  //   TextureReference(size_t width)
  //     : width_{width}, alloc_{LinearAllocs::hipHostMalloc, width * sizeof(TexelType)} {}

  __host__ __device__ TextureReference(TexelType* ptr, size_t width, size_t layers)
      : layers_{layers}, width_{width}, alloc_{ptr} {}


  template <typename F> void Fill(F f) {
    for (auto i = 0u; i < width_; ++i) {
      ptr(0)[i] = f(i);
    }
  }

  // TexelType Fetch1D(int x, hipTextureDesc& tex_desc) {
  //   x = ApplyAddressMode(x, tex_desc.addressMode[0]);

  //   if (x >= width_) {
  //     TexelType ret;
  //     memset(&ret, 0, sizeof(ret));
  //     return ret;
  //   }

  //   return ptr()[x];
  // }

  __host__ __device__ TexelType Tex1DLayered(float x, int layer, hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * width_ : x;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return ApplyAddressMode(floorf(x), layer, tex_desc.addressMode[0]);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, layer, tex_desc.addressMode[0]);
    }
  }

  __host__ __device__ TexelType Tex1D(float x, hipTextureDesc& tex_desc) const {
    return Tex1DLayered(x, 0, tex_desc);
  }

  __host__ __device__ TexelType* ptr(size_t layer) { return alloc_ + layer * width_; }

  __host__ __device__ TexelType* ptr(size_t layer) const { return alloc_ + layer * width_; }

  __host__ __device__ size_t width() const { return width_; }

 private:
  const size_t layers_;
  const size_t width_;
  TexelType* const alloc_;

  __host__ __device__ int ApplyAddressMode(int x, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return (x < width_) * x;
      case hipAddressModeBorder:
        return x;
    }
  }

  __host__ __device__ TexelType ApplyAddressMode(float x, int layer,
                                                 hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp: {
        x = max(min(x, static_cast<float>(width_ - 1)), 0.0f);
        break;
      }
      case hipAddressModeBorder: {
        if (x > width_ - 1 || x < 0.0f) {
          return Zero();
        }
        break;
      }
      case hipAddressModeWrap:
        x /= width_;
        x = x - floorf(x);
        x *= width_;
        break;
      case hipAddressModeMirror: {
        x /= width_;
        const float frac_x = x - floor(x);
        const bool is_reversing = static_cast<int64_t>(floorf(x)) % 2;
        x = is_reversing ? 1.0f - frac_x : frac_x;
        x *= width_;
        x -= (x == truncf(x)) * is_reversing;
        break;
      }
    }

    return ptr(layer)[static_cast<size_t>(x)];
  }

  __host__ __device__ TexelType Zero() const {
    TexelType ret;
    memset(&ret, 0, sizeof(ret));
    return ret;
  }

  __host__ __device__ TexelType LinearFiltering(float x, int layer,
                                                hipTextureAddressMode address_mode) const {
    const auto xB = x - 0.5f;
    const auto i = floorf(xB);
    const auto alpha = FloatToNBitFractional<8>(xB - i);
    const auto T_i0 = ApplyAddressMode(i, layer, address_mode);
    const auto T_i1 = ApplyAddressMode(i + 1, layer, address_mode);
    return Vec4Add(Vec4Scale((1.0f - alpha), T_i0), Vec4Scale(alpha, T_i1));
  }

  template <size_t fractional_bits> __host__ __device__ float FloatToNBitFractional(float x) const {
    const auto fixed_point = static_cast<uint16_t>(roundf(x * (1 << fractional_bits)));
    return static_cast<float>(fixed_point) / static_cast<float>(1 << fractional_bits);
  }
};
