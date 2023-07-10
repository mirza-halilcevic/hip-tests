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
#include <cmath>

#include <hip_test_common.hh>

class TextureGuard {
 public:
  TextureGuard(hipResourceDesc* res_desc, hipTextureDesc* tex_desc) {
    HIP_CHECK(hipCreateTextureObject(&tex_obj_, res_desc, tex_desc, nullptr));
  }

  ~TextureGuard() { static_cast<void>(hipDestroyTextureObject(tex_obj_)); }

  TextureGuard(const TextureGuard&) = delete;
  TextureGuard(TextureGuard&&) = delete;

  TextureGuard& operator=(const TextureGuard&) = delete;
  TextureGuard& operator=(TextureGuard&&) = delete;

  hipTextureObject_t object() const { return tex_obj_; }

 private:
  hipTextureObject_t tex_obj_ = 0;
};

template <typename T> std::enable_if_t<std::is_integral_v<T>, float> NormalizeInteger(const T x) {
  // On the GPU, -1.0 will be returned both  for the minimum value of a signed type and its
  // successor e.g. for char, -1.0 will be returned for both -128 and -127.
  auto xf = std::abs(static_cast<float>(x));
  xf = std::min<float>(xf, std::numeric_limits<T>::max());
  return std::copysign(xf / std::numeric_limits<T>::max(), x);
}

inline std::string AddressModeToString(decltype(hipAddressModeClamp) address_mode) {
  switch (address_mode) {
    case hipAddressModeClamp:
      return "hipAddressModeClamp";
    case hipAddressModeBorder:
      return "hipAddressModeBorder";
    case hipAddressModeWrap:
      return "hipAddressModeWrap";
    case hipAddressModeMirror:
      return "hipAddressModeMirror";
    default:
      throw std::invalid_argument("Invalid hipAddressMode value");
  }
}

inline std::string FilteringModeToString(decltype(hipFilterModePoint) filter_mode) {
  switch (filter_mode) {
    case hipFilterModePoint:
      return "hipFilterModePoint";
    case hipFilterModeLinear:
      return "hipFilterModeLinear";
    default:
      throw std::invalid_argument("Invalid hipFilterMode value");
  }
}