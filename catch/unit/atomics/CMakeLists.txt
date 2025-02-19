# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set(TEST_SRC
    atomicExch.cc
    atomicExch_system.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
    set_source_files_properties(atomicExch_system.cc PROPERTIES COMPILE_FLAGS "-rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS "nvrtc -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
elseif(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS hiprtc)
endif()

# SWDEV-435667: Below 2 tests failed in stress test on 01/12/23
#add_test(NAME Unit_atomicExch_Negative_Parameters
#         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
#         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
#         atomicExch_negative_kernels.cc 40)
#
#add_test(NAME Unit_atomicExch_system_Negative_Parameters
#         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
#         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
#         atomicExch_system_negative_kernels.cc 40)
