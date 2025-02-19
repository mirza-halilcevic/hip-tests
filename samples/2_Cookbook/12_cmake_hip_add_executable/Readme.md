## hip_add_executable ###
This tutorial shows how to use the FindHIP cmake module and create an executable using ```hip_add_executable``` macro.

## Including FindHIP cmake module in the project
Since FindHIP cmake module is not yet a part of the default cmake distribution, ```CMAKE_MODULE_PATH``` needs to be updated to contain the path to FindHIP.cmake.

The simplest approach is to use
```
set(CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip/" ${CMAKE_MODULE_PATH})
find_package(HIP)
```

A more generic solution that allows for a user specified location for the HIP installation would look something like
```
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP)
```

If your project already modifies ```CMAKE_MODULE_PATH```, you will need to append the path to FindHIP.cmake instead of replacing it.

## Using the hip_add_executable macro
FindHIP provides the ```hip_add_executable``` macro that is similar to the ```cuda_add_executable``` macro that is provided by FindCUDA.
The syntax is also similar. The ```hip_add_executable``` macro uses the hipcc wrapper as the compiler.
The macro supports specifying CLANG-specific, NVCC-specific compiler options using the ```CLANG_OPTIONS``` and ```NVCC_OPTIONS``` keywords.
Common options targeting both compilers can be specificed after the ```HIPCC_OPTIONS``` keyword.

## How to build and run:
- Build sample using cmake
```
$ mkdir build; cd build
 # For shared lib of hip rt,
$ cmake ..
 # Or for static lib of hip rt,
$ cmake -DCMAKE_PREFIX_PATH="/opt/rocm/llvm/lib/cmake" ..
$ make
```

- Execute the sample
```
$ ./MatrixTranspose
Device name
PASSED!
```

## More Info:
- [HIP FAQ](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/faq.md)
- [HIP Kernel Language](https://github.com/ROCm/HIP/blob/develop/docs/reference/kernel_language.md)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://github.com/ROCm/HIP/blob/develop/docs/user_guide/hip_porting_guide.md)
- [HIP Terminology](https://github.com/ROCm/HIP/blob/develop/docs/reference/terms.md) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://github.com/ROCm/HIPIFY/blob/amd-staging/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/docs/developer_guide/contributing.md)
