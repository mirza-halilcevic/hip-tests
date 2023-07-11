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

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>

template <typename TexelType> class TextureReference {
 public:
  TextureReference(TexelType* alloc, hipExtent extent, size_t layers)
      : alloc_{alloc}, extent_{extent}, layers_{layers} {}

  // template <typename F> void Fill(F f) {
  //   for (auto i = 0u; i < width_; ++i) {
  //     ptr(0)[i] = f(i);
  //   }
  // }

  TexelType Tex1D(float x, const hipTextureDesc& tex_desc) const {
    return Tex1DLayered(x, 0, tex_desc);
  }

  TexelType Tex2D(float x, float y, const hipTextureDesc& tex_desc) const {
    return Tex2DLayered(x, y, 0, tex_desc);
  }

  TexelType Tex1DLayered(float x, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, layer, tex_desc.addressMode);
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType Tex2DLayered(float x, float y, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), floorf(y), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, y, layer, tex_desc.addressMode);
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType* ptr(size_t layer) { return alloc_ + layer * extent_.width * (extent_.height ?: 1); }

  const TexelType* ptr(size_t layer) const {
    return alloc_ + layer * extent_.width * (extent_.height ?: 1);
  }

  hipExtent extent() const { return extent_; }

 private:
  TexelType* const alloc_;
  const hipExtent extent_;
  const size_t layers_;

  template <size_t fractional_bits> float FloatToNBitFractional(float x) const {
    const auto fixed_point = static_cast<uint16_t>(roundf(x * (1 << fractional_bits)));
    return static_cast<float>(fixed_point) / static_cast<float>(1 << fractional_bits);
  }

  TexelType Zero() const {
    TexelType ret;
    memset(&ret, 0, sizeof(ret));
    return ret;
  }

  float ApplyAddressMode(float coord, size_t dim, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return ApplyClamp(coord, dim);
      case hipAddressModeBorder:
        if (CheckBorder(coord, dim)) {
          return std::numeric_limits<float>::quiet_NaN();
        }
      case hipAddressModeWrap:
        return ApplyWrap(coord, dim);
      case hipAddressModeMirror:
        return ApplyMirror(coord, dim);
      default:
        throw std::invalid_argument("Invalid hipAddressMode value");
    }
  }

  TexelType Sample(float x, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);

    if (std::isnan(x)) {
      return Zero();
    }

    return ptr(layer)[static_cast<size_t>(x)];
  }

  TexelType Sample(float x, float y, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);
    y = ApplyAddressMode(y, extent_.height, address_mode[1]);

    if (std::isnan(x) || std::isnan(y)) {
      return Zero();
    }

    return ptr(layer)[static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)];
  }

  TexelType LinearFiltering(float x, int layer, const hipTextureAddressMode* address_mode) const {
    const auto [xB, i, alpha] = GetLinearFilteringParams(x);

    const auto T_i0 = Sample(i, layer, address_mode);
    const auto T_i1 = Sample(i + 1.0f, layer, address_mode);

    return Vec4Add(Vec4Scale((1.0f - alpha), T_i0), Vec4Scale(alpha, T_i1));
  }

  TexelType LinearFiltering(float x, float y, int layer,
                            const hipTextureAddressMode* address_mode) const {
    const auto [xB, i, alpha] = GetLinearFilteringParams(x);
    const auto [yB, j, beta] = GetLinearFilteringParams(y);

    const auto T_i0j0 = Sample(i, j, layer, address_mode);
    const auto T_i1j0 = Sample(i + 1.0f, j, layer, address_mode);
    const auto T_i0j1 = Sample(i, j + 1.0f, layer, address_mode);
    const auto T_i1j1 = Sample(i + 1.0f, j + 1.0f, layer, address_mode);

    return Vec4Add(
        Vec4Add(Vec4Scale((1.0f - alpha) * (1.0f - beta), T_i0j0),
                Vec4Scale(alpha * (1.0f - beta), T_i1j0)),
        Vec4Add(Vec4Scale((1.0f - alpha) * beta, T_i0j1), Vec4Scale(alpha * beta, T_i1j1)));
  }

  float ApplyClamp(float coord, size_t dim) const {
    return max(min(coord, static_cast<float>(dim - 1)), 0.0f);
  }

  bool CheckBorder(float coord, size_t dim) const { return coord > dim - 1 || coord < 0.0f; }

  float ApplyWrap(float coord, size_t dim) const {
    coord /= dim;
    coord = coord - floorf(coord);
    coord *= dim;

    return coord;
  }

  float ApplyMirror(float coord, size_t dim) const {
    coord /= dim;
    const float frac = coord - floor(coord);
    const bool is_reversing = static_cast<ssize_t>(floorf(coord)) % 2;
    coord = is_reversing ? 1.0f - frac : frac;
    coord *= dim;
    coord -= (coord == truncf(coord)) * is_reversing;

    return coord;
  }

  std::tuple<float, float, float> GetLinearFilteringParams(float coord) const {
    const auto coordB = coord - 0.5f;
    const auto index = floorf(coordB);
    const auto coeff = FloatToNBitFractional<8>(coordB - index);

    return {coordB, index, coeff};
  }
};
