#include <cooperative_groups.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <bitset>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define E(expr)                                                     \
    {                                                               \
        const auto err_code = (expr);                               \
        if (err_code != cudaSuccess) {                              \
            std::cout << cudaGetErrorString(err_code) << std::endl; \
        }                                                           \
    }

namespace cg = cooperative_groups;

template <typename T>
struct vec4_struct {
    using type = void;
};

template <>
struct vec4_struct<unsigned char> {
    using type = uchar4;
};

template <>
struct vec4_struct<unsigned short> {
    using type = ushort4;
};

template <>
struct vec4_struct<unsigned int> {
    using type = uint4;
};

template <>
struct vec4_struct<float> {
    using type = float4;
};

template <typename T>
using vec4 = typename vec4_struct<T>::type;

template <typename T>
auto make_vec4(const T val) {
     vec4<T> vec;
    vec.x = val;
    vec.y = val;
    vec.z = val;
    vec.w = val;

    return vec;
}

template <typename T>
auto make_vec4(const T x, const T y, const T z, const T w) {
     vec4<T> vec;
    vec.x = x;
    vec.y = y;
    vec.z = z;
    vec.w = w;

    return vec;
}

__global__ void kernel(cudaTextureObject_t tex_obj) {
    const auto v = tex1D<float2>(tex_obj, 1.100000023841857910156250000000);
    printf("x:%1.10f, y:%1.10f\n", v.x, v.y);
}

__global__ void kernel2D(cudaTextureObject_t tex_obj) {
    const auto v = tex2D<float2>(tex_obj, -16, -4);
    printf("x:%1.10f, y:%1.10f\n", v.x, v.y);
}

typedef uint16_t fixed_point_t;
#define FIXED_POINT_FRACTIONAL_BITS 8

float fixed_to_double(fixed_point_t input);

fixed_point_t double_to_fixed(float input);

inline float fixed_to_double(fixed_point_t input) {
    return ((float)input / (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

inline fixed_point_t double_to_fixed(float input) {
    return (
        fixed_point_t)(std::round(input * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

int main(int argc, char const *argv[]) {
    const int width = 4;
    const int height = 1;

    std::vector<float2> vec(width * height);
    for (auto i = 0u; i < vec.size(); ++i) {
        vec[i].x = i + 1;
        vec[i].y = i + 1;
    }

    cudaArray_t array;
    const auto desc = cudaCreateChannelDesc<float2>();
    E(cudaMalloc3DArray(&array, &desc, make_cudaExtent(width, 0, 0)));
    const auto spitch = width * sizeof(float2);
    E(cudaMemcpy2DToArray(array, 0, 0, vec.data(), spitch, spitch, height,
                          cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = true;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex;
    E(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));

    kernel<<<1, 1>>>(tex);
    E(cudaDeviceSynchronize());

    const auto x = 0.7;
    const auto xB = 0.7 - 0.5;
    auto alpha = xB - std::floor(xB);
    // alpha = std::trunc(alpha * std::pow(10, 8)) * std::pow(10, -8);
    // auto alpha_raw = *reinterpret_cast<uint32_t *>(&alpha);
    // std::cout << std::bitset<sizeof(alpha_raw) * 8>(alpha_raw) << std::endl;
    // alpha_raw &= 0b0000000'00000000'11111111'11111111'1;
    // alpha = *reinterpret_cast<float *>(&alpha_raw);
    // alpha = fixed_to_double(double_to_fixed(alpha));
    // std::cout << std::fixed << std::setprecision(10) << alpha << std::endl;
    // std::cout << std::fixed << std::setprecision(10)
    //           << ((1.0f - alpha) * 1 + alpha * 2) << std::endl;

    return 0;
}