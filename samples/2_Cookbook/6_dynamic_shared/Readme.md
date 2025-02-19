## Using Dynamic shared memory ###

Earlier we learned how to use static shared memory. In this tutorial, we'll explain how to use the dynamic version of shared memory to improve the performance.

## Introduction:

As we mentioned earlier  that Memory bottlenecks is the main problem why we are not able to get the highest performance, therefore minimizing the latency for memory access plays prominent role in application optimization. In this tutorial, we'll learn how to use dynamic shared memory.

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/ROCm/HIP/blob/develop/docs/how_to_guides/install.md)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the previous tutorial and modify it to learn how to use shared memory.

## Shared Memory

Shared memory is way more faster than that of global and constant memory and accessible to all the threads in the block.

Previously, it was essential to declare dynamic shared memory using the HIP_DYNAMIC_SHARED macro for accuracy, as using static shared memory in the same kernel could result in overlapping memory ranges and data-races.

Now, the HIP-Clang compiler provides support for extern shared declarations, and the HIP_DYNAMIC_SHARED option is no longer required. You may use the standard extern definition:
extern __shared__ type var[];


The other important change is:
```
    hipLaunchKernelGGL(matrixTranspose,
                  dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  sizeof(float)*WIDTH*WIDTH, 0,
                  gpuTransposeMatrix , gpuMatrix, WIDTH);
```
here we replaced 4th parameter with amount of additional shared memory to allocate when launching the kernel.

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./dynamic_shared
Device name Navi 14 [Radeon Pro W5500]
dynamic_shared PASSED!
```
## More Info:
- [HIP FAQ](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/faq.md)
- [HIP Kernel Language](https://github.com/ROCm/HIP/blob/develop/docs/reference/kernel_language.md)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/hip_porting_guide.md)
- [HIP Terminology](https://github.com/ROCm/HIP/blob/develop/docs/reference/terms.md) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://github.com/ROCm/HIPIFY/blob/amd-staging/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/docs/developer_guide/contributing.md)
