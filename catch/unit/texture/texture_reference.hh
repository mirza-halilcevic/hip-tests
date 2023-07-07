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

#include <resource_guards.hh>

template <typename TexelType> class TextureReference {
 public:
  TextureReference(size_t width)
      : width_{width}, host_alloc_{LinearAllocs::hipHostMalloc, width * sizeof(TexelType)} {}

  template <typename F> void Fill(F f) {
    for (auto i = 0u; i < width_; ++i) {
      ptr()[i] = f(i);
    }
  }

  TexelType Fetch1D(int x, hipTextureDesc& tex_desc) {
    x = ApplyAddressMode(x, tex_desc.addressMode[0]);

    if (x >= width_) {
      TexelType ret;
      memset(&ret, 0, sizeof(ret));
      return ret;
    }

    return ptr()[x];
  }

  TexelType Tex1D(float x, hipTextureDesc& tex_desc) {
    x = tex_desc.normalizedCoords ? x * width_ : x;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return ApplyAddressMode(std::floor(x), tex_desc.addressMode[0]);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      throw "Idiot";
    } else {
      throw "Ded";
    }
  }

  TexelType* ptr() { return host_alloc_.ptr(); }

  TexelType* ptr() const { return host_alloc_.ptr(); }

  size_t width() const { return width_; }

 private:
  const size_t width_;
  LinearAllocGuard<TexelType> host_alloc_;

  int ApplyAddressMode(int x, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return (x < width_) * x;
      case hipAddressModeBorder:
        return x;
      default:
        throw "Ded";
    }
  }

  TexelType ApplyAddressMode(float x, hipTextureAddressMode address_mode) const {
    const auto normalized_width = 1.0f - 1.0f / width_;
    switch (address_mode) {
      case hipAddressModeClamp: {
        x = std::max(std::min<float>(x, width_ - 1), 0.0f);
        break;
      }
      case hipAddressModeBorder: {
        if (x > width_ - 1 || x < 0.0f) {
          return Zero();
        }
        break;
      }
      case hipAddressModeWrap:
        x = x / width_;
        x = x - std::floor(x);
        x = x * width_;
        break;
      case hipAddressModeMirror: {
        x = x / width_;
        const float frac_x = x - std::floor(x);
        const bool is_reversing = static_cast<size_t>(std::floor(x)) % 2;
        x = is_reversing ? 1.0f - frac_x : frac_x;
        x = x * width_;
        const auto offset = (x == std::trunc(x)) * is_reversing;
        x = x - offset;
        break;
      }
      default:
        throw "Ded";
    }

    return ptr()[static_cast<size_t>(x)];
  }

  float DenormalizeCoordinate(float x, bool normalized_coords) const {
    return normalized_coords ? x * width_ : x;
  }

  TexelType Zero() const {
    TexelType ret;
    memset(&ret, 0, sizeof(ret));
    return ret;
  }

  TexelType ApplyFiltering(float x, decltype(hipFilterModeLinear) filter_mode) {
    switch (filter_mode) {
      case hipFilterModePoint:
        return ptr()[static_cast<size_t>(x)];
      case hipFilterModeLinear: {
        float xB = x - 0.5f;
        xB = std::max(xB, 0.0f);
        int i = std::floor(xB);
        float alfa = xB - i;
        return Vec4Add(Vec4Scale((1 - alfa), ptr()[i]), Vec4Scale(alfa, ptr()[i + 1]));
      }
      default:
        throw "Ded";
    }
  }
};
