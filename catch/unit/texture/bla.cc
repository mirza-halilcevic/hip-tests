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


template <typename TexelType>
__global__ void tex1DfetchKernel(TexelType* const out, hipTextureObject_t tex_obj) {
  const auto tid = cg::this_grid().thread_rank();
  out[tid] = tex1Dfetch<TexelType>(tex_obj, tid);
}

TEST_CASE("tex1Dfetch") {
  using T = unsigned int;
  using TexelType = vec4<T>;

  const auto width = 1024;

  std::vector<TexelType> tex_alloc_h(width);
  for (auto i = 0u; i < tex_alloc_h.size(); ++i) {
    SetVec4<T>(tex_alloc_h[i], i);
  }

  LinearAllocGuard<TexelType> tex_alloc_d(LinearAllocs::hipMalloc, width * sizeof(TexelType));
  HIP_CHECK(hipMemcpy(tex_alloc_d.ptr(), tex_alloc_h.data(), tex_alloc_h.size() * sizeof(TexelType),
                      hipMemcpyHostToDevice));

  LinearAllocGuard<TexelType> out_alloc_d(LinearAllocs::hipMalloc, width * sizeof(TexelType));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<TexelType>();
  res_desc.res.linear.sizeInBytes = width * sizeof(TexelType);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = hipAddressModeClamp;
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.normalizedCoords = false;

  TextureGuard tex(&res_desc, &tex_desc);
  tex1DfetchKernel<TexelType><<<1, width>>>(out_alloc_d.ptr(), tex.object());

  std::vector<TexelType> out_alloc_h(width);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), width * sizeof(TexelType),
                      hipMemcpyDeviceToHost));

  for (auto i = 0u; i < out_alloc_h.size(); ++i) {
    REQUIRE(tex_alloc_h[i].x == out_alloc_h[i].x);
    REQUIRE(tex_alloc_h[i].y == out_alloc_h[i].y);
    REQUIRE(tex_alloc_h[i].z == out_alloc_h[i].z);
    REQUIRE(tex_alloc_h[i].w == out_alloc_h[i].w);
  }
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
