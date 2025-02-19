## Using Pragma unroll ###

In this tutorial, we'll explain how to use #pragma unroll to improve the performance.

## Introduction:

Loop unrolling optimization hints can be specified with #pragma unroll and #pragma nounroll. The pragma is placed immediately before a for loop.
Specifying #pragma unroll without a parameter directs the loop unroller to attempt to fully unroll the loop if the trip count is known at compile time and attempt to partially unroll the loop if the trip count is not known at compile time.

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/ROCm/HIP/blob/develop/docs/how_to_guides/install.md)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

For this tutorial we will be using an example which sums up the row of a 2D matrix and writes it in a 1D array.

In this tutorial, we'll use `#pragma unroll`. In the same sourcecode, we used for gpuMatrixRowSum. We'll add it just before the for loop as following:

```
#pragma unroll
for (int i = 0; i < width; i++) {
    output[index] += input[index * width + i]
}
```

Specifying the optional parameter, #pragma unroll value, directs the unroller to unroll the loop value times. Be careful while using it.
Specifying #pragma nounroll indicates that the loop should not be unroll. #pragma unroll 1 will show the same behaviour.

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./unroll
Device name
PASSED
```
## requirement for nvidia
please make sure you have a 3.0 or higher compute capable device in order to use warp shfl operations and add `-gencode arch=compute=30, code=sm_30` nvcc flag in the Makefile while using this application.

## More Info:
- [HIP FAQ](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/faq.md)
- [HIP Kernel Language](https://github.com/ROCm/HIP/blob/develop/docs/reference/kernel_language.md)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/hip_porting_guide.md)
- [HIP Terminology](https://github.com/ROCm/HIP/blob/develop/docs/reference/terms.md) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://github.com/ROCm/HIPIFY/blob/amd-staging/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/docs/developer_guide/contributing.md)
