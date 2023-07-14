/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes tests for the graph management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DeviceTest Device Management
 * @{
 * This section describes tests for device management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup EventTest Event Management
 * @{
 * This section describes tests for the event management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ErrorTest Error Handling
 * @{
 * This section describes tests for the error handling functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DriverTest Initialization and Version
 * @{
 * This section describes tests for the initialization and version functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ModuleTest Module Management
 * @{
 * This section describes tests for the module management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ExecutionTest Execution Control
 * @{
 * This section describes tests for the execution control functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamTest Stream Management
 * @{
 * This section describes tests for the stream management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamMTest Stream Memory Operations
 * @{
 * This section describes tests for the Stream Memory Wait and Write functions of HIP runtime API.
 */

// Adding dummy Test Cases that are in the form of function macros/templates and are
// not possible to generate with Doxygen.

/**
 * @addtogroup hipStreamWaitValue32 hipStreamWaitValue32
 * @{
 * @ingroup StreamMTest
 * `hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value,
 * unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF))` -
 * Enqueues a wait command to the stream, all operations enqueued on this stream after this, will
 * not execute until the defined wait condition is true.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_And") {}
TEST_CASE("Unit_hipStreamValue_Negative_InvalidMemory") {}
/**
 * End doxygen group hipStreamWaitValue32.
 * @}
 */

/**
 * @addtogroup hipStreamWaitValue64 hipStreamWaitValue64
 * @{
 * @ingroup StreamMTest
 * `hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value,
 * unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF))` -
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipStreamValue_Negative_InvalidFlag
 *  - @ref Unit_hipStreamValue_Negative_InvalidMemory
 *  - @ref Unit_hipStreamValue_Negative_UninitializedStream
 */

/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Gte_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Gte_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Gte_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Gte_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_And") {}
/**
 * End doxygen group hipStreamWaitValue64.
 * @}
 */

/**
 * @addtogroup hipStreamWriteValue64 hipStreamWriteValue64
 * @{
 * @ingroup StreamMTest
 * `hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags)` -
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipStreamValue_Write
 *  - @ref Unit_hipStreamValue_Negative_InvalidMemory
 *  - @ref Unit_hipStreamValue_Negative_UninitializedStream
 * @}
 */

/**
 * End doxygen group StreamMTest.
 * @}
 */

/**
 * @defgroup StreamOTest Ordered Memory Allocator
 * @{
 * This section describes the tests for Stream Ordered Memory Allocator functions of HIP runtime
 * API.
 * @}
 */

/**
 * @defgroup PeerToPeerTest PeerToPeer Device Memory Access
 * @{
 * This section describes tests for the PeerToPeer device memory access functions of HIP runtime
 * API.
 * @warning PeerToPeer support is experimental.
 * @}
 */

/**
 * @defgroup ContextTest Context Management
 * @{
 * This section describes tests for the context management functions of HIP runtime API.
 * @warning All Context Management APIs are **deprecated** and shall not be implemented.
 * @}
 */

/**
 * @defgroup TextureTest Texture Management
 * @{
 * This section describes tests for the texture management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup OccupancyTest Occupancy
 * @{
 * This section describes tests for the occupancy functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup GLTest Interop
 * @{
 * This section describes tests for the GL interop functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup MemoryTest Memory Management
 * @{
 * This section describes tests for the memory management functions of HIP runtime API.
 */

/**
 * @addtogroup hipMemset hipMemset
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemset") {}
/**
 * End doxygen group hipMemset.
 * @}
 */

/**
 * @addtogroup hipMemsetD32 hipMemsetD32
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") {}
/**
 * End doxygen group hipMemsetD32.
 * @}
 */

/**
 * @addtogroup hipMemsetD16 hipMemsetD16
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") {}
/**
 * End doxygen group hipMemsetD16.
 * @}
 */

/**
 * @addtogroup hipMemsetD8 hipMemsetD8
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") {}
/**
 * End doxygen group hipMemsetD8.
 * @}
 */

/**
 * End doxygen group MemoryTest.
 * @}
 */

/**
 * @defgroup MemoryMTest Managed Memory
 * @{
 * This section describes tests for the managed memory management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup PeerToPeerTest PeerToPeer Device Memory Access
 * @{
 *  @warning PeerToPeer support is experimental.
 *  This section describes tests for the PeerToPeer device memory access functions of HIP runtime
 * API.
 * @}
 */

/**
 * @defgroup VirtualTest Virtual Memory Management
 * @{
 * This section describes tests for the virtual memory management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
 */

/**
 * @addtogroup atomicAdd atomicAdd
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicMin with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAdd_Negative_Parameters") {}
/**
 * End doxygen group atomicAdd.
 * @}
 */

/**
 * @addtogroup atomicSub atomicSub
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicSub with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicSub_Negative_Parameters") {}
/**
 * End doxygen group atomicSub.
 * @}
 */

/**
 * @addtogroup atomicInc atomicInc
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicInc with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicInc_Negative_Parameters") {}
/**
 * End doxygen group atomicInc.
 * @}
 */

/**
 * @addtogroup atomicDec atomicDec
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicDec with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicDec_Negative_Parameters") {}
/**
 * End doxygen group atomicDec.
 */

/**
 * @addtogroup atomicExch atomicExch
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicExch with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicExch_Negative_Parameters") {}
/**
 * End doxygen group atomicExch.
 * @}
 */

/**
 * @addtogroup atomicMin atomicMin
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicMin with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMin_Negative_Parameters") {}
/**
 * End doxygen group atomicMin.
 * @}
 */

/**
 * @addtogroup atomicMax atomicMax
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicMax with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMax_Negative_Parameters") {}
/**
 * End doxygen group atomicMax.
 * @}
 */

/**
 * @addtogroup atomicAnd atomicAnd
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicAnd with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Negative_Parameters") {}
/**
 * End doxygen group atomicAnd.
 * @}
 */

/**
 * @addtogroup atomicOr atomicOr
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicOr with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Negative_Parameters") {}
/**
 * End doxygen group atomicOr.
 * @}
 */

/**
 * @addtogroup atomicXor atomicXor
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicXor with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Negative_Parameters") {}
/**
 * End doxygen group atomicXor.
 * @}
 */

/**
 * End doxygen group AtomicsTest.
 * @}
 */

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 * @}
 */

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 */

/**
 * @addtogroup launch_bounds launch_bounds
 * @{
 * @ingroup DeviceLanguageTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Compiles kernels that are not created appropriately:
 *      - Maximum number of threads is 0
 *      - Maximum number of threads is not integer value
 *      - Mimimum number of warps is not integer value
 *    -# Expected output: compiler error
 * Test source
 * ------------------------
 *  - unit/launch_bounds/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_CompilerError") {}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Compiles kernels that are not created appropriately:
 *      - Maximum number of threads is negative
 *      - Mimimum number of warps is negative
 *  - Validates handling of invalid arguments:
 *    -# Expected output: parse error
 * Test source
 * ------------------------
 *  - unit/launch_bounds/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_ParseError") {}

/**
 * End doxygen group launch_bounds.
 * @}
 */

/**
 * @addtogroup static_assert static_assert
 * @{
 * @ingroup DeviceLanguageTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles kernels with static_assert calls:
 *    -# Expected that static_assert passes and compilation is successful.
 *    -# Expected that static_assert fails and compilation has errors.
 * Test source
 * ------------------------
 *  - unit/assertion/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_StaticAssert_Positive_Basic") {}

/**
 * Test Description
 * ------------------------
 *  - Passes invalidly formed expressions to static_assert calls.
 *  - Uses expressions that are not constexpr and values that are not known during compilation.
 * Test source
 * ------------------------
 *  - unit/assertion/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_StaticAssert_Negative_Basic")

/**
 * End doxygen group static_assert.
 * @}
 */

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */

/**
 * @defgroup ShflTest warp shuffle function Management
 * @{
 * This section describes the warp shuffle types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup SyncthreadsTest Synchronization Functions
 * @{
 * This section describes tests for Synchronization Functions.
 * @}
 */

/**
 * @defgroup ThreadfenceTest Memory Fence Functions
 * @{
 * This section describes tests for Memory Fence Functions.
 */

/**
 * @defgroup MathTest Math Device Functions
 * @{
 * This section describes tests for device math functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup VectorTypeTest Vector types
 * @{
 * This section describes tests for the Vector type functions and operators.
 */

/**
 * @addtogroup make_vector make_vector
 * @{
 * @ingroup VectorTypeTest
 */

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Negate (-) operation applied on the unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_NegateUnsigned_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Bitwise operations applied on the float vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_BitwiseFloat_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Bitwise operations applied on the double vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_BitwiseDouble_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 1D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign1D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 2D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign2D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 3D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign3D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 4D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign4D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 1D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned1D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 2D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned2D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 3D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned3D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 4D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned4D_Negative_Parameters") {}

/**
 * End doxygen group make_vector.
 * @}
 */

/**
 * End doxygen group VectorTypeTest.
 * @}
 */
