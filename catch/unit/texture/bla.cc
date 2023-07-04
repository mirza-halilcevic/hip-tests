#include <vector>

#include <hip_test_common.hh>
#include <cooperative_groups.h>
#include <resource_guards.hh>

#include "vec4.hh"

namespace cg = cooperative_groups;

class TextureGuard {
 public:
  TextureGuard(hipResourceDesc* res_desc, hipTextureDesc* tex_desc) {
    HIP_CHECK(hipCreateTextureObject(&tex_obj_, res_desc, tex_desc, nullptr));
  }

  ~TextureGuard() { static_cast<void>(hipDestroyTextureObject(tex_obj_)); }

  TextureGuard(TextureGuard&&) = delete;
  TextureGuard(const TextureGuard&) = delete;

  hipTextureObject_t object() const { return tex_obj_; }

 private:
  hipTextureObject_t tex_obj_ = 0;
};

template <typename TexelType, typename UnderlyingType> class TextureReference {
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
    return ptr()[x];
  }

  TexelType* ptr() { return host_alloc_.ptr(); }

 private:
  const size_t width_;
  LinearAllocGuard<TexelType> host_alloc_;

  int ApplyAddressMode(int x, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return std::min<int>(x, width_);
      case hipAddressModeBorder:
        return (x < width_) * x;
      default:
        throw "Ded";
    }
  }
};

template <typename T> std::enable_if_t<std::is_integral_v<T>, float> NormalizeInteger(const T x) {
  return x >= static_cast<T>(0) ? static_cast<float>(x) / std::numeric_limits<T>::max()
                                : -static_cast<float>(x) / std::numeric_limits<T>::min();
}

template <typename TexelType>
__global__ void tex1DfetchKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;

  out[tid] = tex1Dfetch<TexelType>(tex_obj, tid);
}

TEMPLATE_TEST_CASE("tex1Dfetch", "", char, int, unsigned int, float) {
  using T = TestType;
  using TexelType = vec4<T>;

  const auto width = 1024;

  TextureReference<TexelType, T> tex_h(width);
  // TODO - Need some negative values for signed types.
  tex_h.Fill([](size_t x) { return MakeVec4<T>(x); });

  LinearAllocGuard<TexelType> tex_alloc_d(LinearAllocs::hipMalloc, width * sizeof(TexelType));
  HIP_CHECK(
      hipMemcpy(tex_alloc_d.ptr(), tex_h.ptr(), width * sizeof(TexelType), hipMemcpyHostToDevice));

  LinearAllocGuard<TexelType> out_alloc_d(LinearAllocs::hipMalloc, 2 * width * sizeof(TexelType));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<TexelType>();
  res_desc.res.linear.sizeInBytes = width * sizeof(TexelType);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));

  SECTION("Clamp") {
    tex_desc.addressMode[0] = hipAddressModeClamp;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeElementType;
    tex_desc.normalizedCoords = false;

    TextureGuard tex(&res_desc, &tex_desc);
    tex1DfetchKernel<TexelType><<<2, width>>>(out_alloc_d.ptr(), width * 2, tex.object());

    std::vector<TexelType> out_alloc_h(2 * width);

    HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), 2 * width * sizeof(TexelType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for (auto i = 0u; i < out_alloc_h.size(); ++i) {
      const auto ref_val = tex_h.Fetch1D(i, tex_desc);
      REQUIRE(ref_val.x == out_alloc_h[i].x);
      REQUIRE(ref_val.y == out_alloc_h[i].y);
      REQUIRE(ref_val.z == out_alloc_h[i].z);
      REQUIRE(ref_val.w == out_alloc_h[i].w);
    }
  }

  SECTION("Border") {
    tex_desc.addressMode[0] = hipAddressModeBorder;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeElementType;
    tex_desc.normalizedCoords = false;

    TextureGuard tex(&res_desc, &tex_desc);
    tex1DfetchKernel<TexelType><<<2, width>>>(out_alloc_d.ptr(), width * 2, tex.object());

    std::vector<TexelType> out_alloc_h(2 * width);

    HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), 2 * width * sizeof(TexelType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for (auto i = 0u; i < out_alloc_h.size(); ++i) {
      const auto ref_val = tex_h.Fetch1D(i, tex_desc);
      REQUIRE(ref_val.x == out_alloc_h[i].x);
      REQUIRE(ref_val.y == out_alloc_h[i].y);
      REQUIRE(ref_val.z == out_alloc_h[i].z);
      REQUIRE(ref_val.w == out_alloc_h[i].w);
    }
  }
}

TEMPLATE_TEST_CASE("tex1Dfetch_normalized", "", char, unsigned char) {
  using T = TestType;
  using TexelType = vec4<T>;

  const auto width = 1024;

  TextureReference<TexelType, T> tex_h(width);
  // TODO - Need some negative values for signed types.
  tex_h.Fill([](size_t x) { return MakeVec4<T>(x); });

  LinearAllocGuard<TexelType> tex_alloc_d(LinearAllocs::hipMalloc, width * sizeof(TexelType));
  HIP_CHECK(
      hipMemcpy(tex_alloc_d.ptr(), tex_h.ptr(), width * sizeof(TexelType), hipMemcpyHostToDevice));

  LinearAllocGuard<vec4<float>> out_alloc_d(LinearAllocs::hipMalloc,
                                            2 * width * sizeof(vec4<float>));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<TexelType>();
  res_desc.res.linear.sizeInBytes = width * sizeof(TexelType);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));

  SECTION("Clamp") {
    tex_desc.addressMode[0] = hipAddressModeClamp;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeNormalizedFloat;
    tex_desc.normalizedCoords = false;

    TextureGuard tex(&res_desc, &tex_desc);
    tex1DfetchKernel<vec4<float>><<<2, width>>>(out_alloc_d.ptr(), width * 2, tex.object());

    std::vector<vec4<float>> out_alloc_h(2 * width);

    HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), 2 * width * sizeof(vec4<float>),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for (auto i = 0u; i < out_alloc_h.size(); ++i) {
      INFO("i: " << i);
      const auto ref_val = tex_h.Fetch1D(i, tex_desc);
      CHECK(NormalizeInteger(ref_val.x) == out_alloc_h[i].x);
      // REQUIRE(ref_val.y == out_alloc_h[i].y);
      // REQUIRE(ref_val.z == out_alloc_h[i].z);
      // REQUIRE(ref_val.w == out_alloc_h[i].w);
    }
  }

  // SECTION("Border") {
  //   tex_desc.addressMode[0] = hipAddressModeBorder;
  //   tex_desc.filterMode = hipFilterModePoint;
  //   tex_desc.readMode = hipReadModeNormalizedFloat;
  //   tex_desc.normalizedCoords = false;

  //   TextureGuard tex(&res_desc, &tex_desc);
  //   tex1DfetchKernel<vec4<float>><<<2, width>>>(out_alloc_d.ptr(), width * 2, tex.object());

  //   std::vector<vec4<float>> out_alloc_h(2 * width);

  //   HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), 2 * width * sizeof(TexelType),
  //                       hipMemcpyDeviceToHost));
  //   HIP_CHECK(hipDeviceSynchronize());

  //   for (auto i = 0u; i < out_alloc_h.size(); ++i) {
  //     const auto ref_val = tex_h.Fetch1D(i, tex_desc);
  //     REQUIRE(ref_val.x == out_alloc_h[i].x);
  //     REQUIRE(ref_val.y == out_alloc_h[i].y);
  //     REQUIRE(ref_val.z == out_alloc_h[i].z);
  //     REQUIRE(ref_val.w == out_alloc_h[i].w);
  //   }
  // }
}

template <typename Vec> __global__ void kernel(hipTextureObject_t tex_obj) {
  const auto v = tex1D<Vec>(tex_obj, 1);
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

  hipArray_t hip_arr;
  HIP_CHECK(hipMallocArray(&hip_arr, &channel_desc, width, height));

  const size_t spitch = width * sizeof(T);
  HIP_CHECK(hipMemcpy2DToArray(hip_arr, 0, 0, h_data.data(), spitch, spitch, height,
                               hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = hip_arr;

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = hipAddressModeWrap;
  tex_desc.addressMode[1] = hipAddressModeWrap;
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.normalizedCoords = 0;

  hipTextureObject_t tex_obj = 0;
  HIP_CHECK(hipCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

  kernel<T><<<1, 1>>>(tex_obj);
  HIP_CHECK(hipDeviceSynchronize());
}
