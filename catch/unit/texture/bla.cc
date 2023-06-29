#include <vector>

#include <hip_test_common.hh>

#include "vec4.hh"


template<typename Vec>
__global__ void kernel(hipTextureObject_t tex_obj) {
  const auto v = tex1DLod<Vec>(tex_obj, 1, 0);
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
