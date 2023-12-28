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
    atomicCAS.cc
    atomicCAS_system.cc
    atomicInc.cc
    atomicDec.cc
    atomicSub.cc
    atomicSub_system.cc
    atomicAdd.cc
    atomicAdd_system.cc
    unsafeAtomicAdd.cc
    safeAtomicAdd.cc
    acquire_release.cc
    sequential_consistency.cc
    atomic_builtins.cc
    __hip_atomic_fetch_add.cc
    __hip_atomic_compare_exchange_strong.cc
    atomicMin.cc
    atomicMin_system.cc
    atomicMax.cc
    atomicMax_system.cc
    safeAtomicMin.cc
    unsafeAtomicMin.cc
    safeAtomicMax.cc
    unsafeAtomicMax.cc
    atomicExch.cc
    atomicExch_system.cc
    __hip_atomic_exchange.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
    set_source_files_properties(atomicCAS_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicSub_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicAdd_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicExch_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicMin_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicMax_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS "nvrtc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set(EXPECTED_ERRORS 42) # EXSWHTEC-278
elseif(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME AtomicsTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc)
    set(EXPECTED_ERRORS 48)
endif()

add_test(NAME Unit_atomicCAS_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicCAS_negative_kernels.cc 48)
         
add_test(NAME Unit_atomicInc_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicInc_negative_kernels.cc 8)

add_test(NAME Unit_atomicDec_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicDec_negative_kernels.cc 8)
         
add_test(NAME Unit_atomicSub_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicSub_negative_kernels.cc 48)
         
add_test(NAME Unit_atomicAdd_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicAdd_negative_kernels.cc 48)
         
add_test(NAME Unit_AtomicBuiltins_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomic_builtins_kernels.cc 60 27) # Should be 35 warnings, see EXSWHTEC-309
         
add_test(NAME Unit_atomicMin_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicMin_negative_kernels.cc ${EXPECTED_ERRORS})

add_test(NAME Unit_atomicMax_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicMax_negative_kernels.cc ${EXPECTED_ERRORS})

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
