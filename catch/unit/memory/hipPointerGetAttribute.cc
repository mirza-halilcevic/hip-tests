/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <string>

/**
 * @addtogroup hipPointerGetAttribute hipPointerGetAttribute
 * @{
 * @ingroup MemoryTest
 * `hipPointerGetAttribute(void* data, hipPointer_attribute attribute,
 * hipDeviceptr_t ptr)` -
 * Returns information about the specified pointer.[BETA]
 */

static constexpr auto NUM_W{16};
static constexpr auto NUM_H{16};
static constexpr size_t N {10};
#define INT_VAL 10
#define VAL_DATA 99
static __global__ void var_update(int* data) {
  for (unsigned int i = 0; i < N; i++) {
     data[i] = VAL_DATA;
  }
}

/**
 * Test Description
 * ------------------------
 *  - Allocate memory using different Allocation APIs and check whether
 *    correct memory type and device oridinal are returned.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_MemoryTypes") {
  HIP_CHECK(hipSetDevice(0));
  size_t pitch_A;
  size_t width{NUM_W * sizeof(char)};
  unsigned int datatype;
  SECTION("Malloc Pitch Allocation") {
    char *A_d;
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
          &pitch_A, width, NUM_H));
    HIP_CHECK(hipPointerGetAttribute(&datatype,
              HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
              reinterpret_cast<hipDeviceptr_t>(A_d)));
#if HT_NVIDIA
    REQUIRE(datatype == CU_MEMORYTYPE_DEVICE);
#else
    REQUIRE(datatype == hipMemoryTypeDevice);
#endif
  }
#if HT_AMD
  SECTION("Malloc Array Allocation") {
    hipArray *B_d;
    hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
    HIP_CHECK(hipMallocArray(&B_d, &desc, NUM_W, NUM_H, hipArrayDefault));
    HIP_CHECK(hipPointerGetAttribute(&datatype,
                                     HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                     reinterpret_cast<hipDeviceptr_t>(B_d)));
#if HT_NVIDIA
    REQUIRE(datatype == CU_MEMORYTYPE_ARRAY);
#else
    REQUIRE(datatype == hipMemoryTypeArray);
#endif
    HIP_CHECK(hipFreeArray(B_d));
  }

  SECTION("Malloc 3D Array Allocation") {
    int width = 10, height = 10, depth = 10;
    hipArray *arr;

    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(float)*8,
        0, 0, 0, hipChannelFormatKindFloat);
    HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height,
            depth), hipArrayDefault));
    HIP_CHECK(hipPointerGetAttribute(&datatype,
                                     HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                     reinterpret_cast<hipDeviceptr_t>(arr)));
#if HT_NVIDIA
    REQUIRE(datatype == CU_MEMORYTYPE_ARRAY);
#else
    REQUIRE(datatype == hipMemoryTypeArray);
#endif
    HIP_CHECK(hipFreeArray(arr));
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Allocates device variable and gets the pointer attribute.
 *  - Launches kernel with the device variable.
 *  - Verifies that the pointer attribute variable is getting updated.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_KernelUpdation") {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  Nbytes = N * sizeof(int);
  int* A_d, *A_h;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  hipDeviceptr_t data = 0;
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  for (unsigned int i = 0; i < N; i++) {
    A_h[i] = INT_VAL;
  }
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipPointerGetAttribute(&data,
                                   HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  hipLaunchKernelGGL(var_update, dim3(1), dim3(1), 0, 0,
                     reinterpret_cast<int *>(data));
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (unsigned int i = 0; i < N; i++) {
     REQUIRE(A_h[i] == VAL_DATA);
  }
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - Verifies the pointer info of device variable
 *    from peer GPU device.
 *  - Validates the memory type and device ordinal in peer GPU.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_PeerGPU") {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  Nbytes = N * sizeof(int);
  int* A_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  unsigned int data = 0;
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipPointerGetAttribute(&data,
                HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                reinterpret_cast<hipDeviceptr_t>(A_d)));
#if HT_NVIDIA
      REQUIRE(data == CU_MEMORYTYPE_DEVICE);
#else
      REQUIRE(data == hipMemoryTypeDevice);
#endif
      HIP_CHECK(hipPointerGetAttribute(&data,
                HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                reinterpret_cast<hipDeviceptr_t>(A_d)));
      REQUIRE(data == 0);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
  HIP_CHECK(hipFree(A_d));
}

/**
 * Test Description
 * ------------------------
 *  - Allocate device memory and get the buffer ID.
 *  - DeAllocate and Allocate the memory again and ensure
 *    that the buffer ID is unique.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_BufferID") {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  Nbytes = N * sizeof(int);
  int* A_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  unsigned int bufid1, bufid2;
  HIP_CHECK(hipPointerGetAttribute(&bufid1,
            HIP_POINTER_ATTRIBUTE_BUFFER_ID,
            reinterpret_cast<hipDeviceptr_t>(A_d)));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipPointerGetAttribute(&bufid2,
            HIP_POINTER_ATTRIBUTE_BUFFER_ID,
            reinterpret_cast<hipDeviceptr_t>(A_d)));
  REQUIRE(bufid1 != bufid2);
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Allocate host memory and get the device ordinal.
 *  - Ensure that it matches with CUDA result.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_HostDeviceOrdinal") {
  size_t Nbytes = 0;
  Nbytes = N * sizeof(int);
  int* A_h;
  unsigned int data = 0, data1 = 0;
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  for (unsigned int i = 0; i < N; i++) {
    A_h[i] = INT_VAL;
  }
  REQUIRE(hipPointerGetAttribute(&data,
          HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
         reinterpret_cast<hipDeviceptr_t>(A_h)) == hipErrorInvalidValue);
  REQUIRE(hipPointerGetAttribute(&data1,
                                 HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                 reinterpret_cast<hipDeviceptr_t>(A_h))
                                 == hipErrorInvalidValue);
  free(A_h);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Allocate managed memory with different flags.
 *  - Get attribute for the mapped attribute.
 *  - Verify behaviour.
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_MappedMem") {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  Nbytes = N * sizeof(int);
  int* A_d, *A_h;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<int*>(malloc(Nbytes));
  for (unsigned int i = 0; i < N; i++) {
    A_h[i] = INT_VAL;
  }
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  int *ptr1 = 0, *ptr2 = 0;
  unsigned int hostMalloc_mapped = 0;
  unsigned int mallocManaged = 0;
  HIP_CHECK(hipHostMalloc(&ptr1, Nbytes, hipHostMallocMapped));
  HIP_CHECK(hipMallocManaged(&ptr2, Nbytes, hipMemAttachGlobal));
  HIP_CHECK(hipPointerGetAttribute(&hostMalloc_mapped,
                                   HIP_POINTER_ATTRIBUTE_MAPPED,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  HIP_CHECK(hipPointerGetAttribute(&mallocManaged,
                                   HIP_POINTER_ATTRIBUTE_MAPPED,
                                   reinterpret_cast<hipDeviceptr_t>(ptr2)));
  REQUIRE(hostMalloc_mapped == 1);
  REQUIRE(mallocManaged == 1);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipHostFree(ptr1));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the attribute is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When address pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When required start address of host pointer
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memory is deallocated before getting attribute
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When host pointer attribute is required from the device
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When buffer ID attribute is required from the host pointer
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When invalid attribute (-1) is passed
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When getting attributes not supported by the device
 *      - Platform specific (AMD)
 *      - Expected output: return `hipErrorNotSupported`
 * Test source
 * ------------------------
 *  - unit/memory/hipPointerGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipPointerGetAttribute_Negative") {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  constexpr size_t N {100};
  Nbytes = N * sizeof(char);
  char* A_d;
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  hipDeviceptr_t data = 0;
  char *A_h;
  A_h = reinterpret_cast<char*>(malloc(Nbytes));
  SECTION("Pass nullptr to data") {
    REQUIRE(hipPointerGetAttribute(nullptr,
          HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
          reinterpret_cast<hipDeviceptr_t>(A_d))
        == hipErrorInvalidValue);
  }
  SECTION("Pass nullptr to device attribute") {
#if HT_AMD
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
            nullptr) == hipErrorInvalidValue);
#else
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
          reinterpret_cast<hipDeviceptr_t>(nullptr)) == hipErrorInvalidValue);
#endif
  }
  SECTION("DeAllocateMem and get the pointer info") {
    char *B_d;
    HIP_CHECK(hipMalloc(&B_d, Nbytes));
    HIP_CHECK(hipFree(B_d));
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
          reinterpret_cast<hipDeviceptr_t>(B_d)) == hipErrorInvalidValue);
  }
  SECTION("Get Start address of host pointer") {
    char *A_h;
    A_h = reinterpret_cast<char*>(malloc(Nbytes));
    REQUIRE(hipPointerGetAttribute(&data,
          HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
          reinterpret_cast<hipDeviceptr_t>(A_h)) == hipErrorInvalidValue);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_HOST_POINTER to device pointer") {
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_HOST_POINTER,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorInvalidValue);
  }
  SECTION("Pass BUFFER_ID attribute to host pointer") {
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
          reinterpret_cast<hipDeviceptr_t>(A_h))
        == hipErrorInvalidValue);
  }
  SECTION("Pass invalid attribute") {
    REQUIRE(hipPointerGetAttribute(&data, static_cast<hipPointer_attribute>(-1),
                                   reinterpret_cast<hipDeviceptr_t>(A_h)) == hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Pass HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE"
      "not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data,
          HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_CONTEXT not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_CONTEXT,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_P2P_TOKENS  not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE"
      "not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data,
          HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
  SECTION("Pass HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES"
      "not supported by HIP") {
    REQUIRE(hipPointerGetAttribute(&data,
          HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
          reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorNotSupported);
  }
#endif
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
