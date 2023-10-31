# Common Tests - Test independent of all platforms
set(TEST_SRC
  thread_block_tile.cc
  hipCGGridGroupType.cc
  hipCGThreadBlockTileType.cc
  hipCGThreadBlockTileTypeShfl.cc
  hipLaunchCooperativeKernel.cc
  hipLaunchCooperativeKernelMultiDevice.cc
  hipCGCoalescedGroupType.cc
  hipCGCoalescedGroupTypeShfl_down.cc
  hipCGCoalescedGroupTypeShfl_up.cc
  hipCGCoalescedGroupTypeShfl.cc
  thread_block.cc
  grid_group.cc
  coalesced_group.cc
  coalesced_group_tiled_partition.cc
  multi_grid_group.cc
  hipCGThreadBlockType_old.cc
  hipCGMultiGridGroupType_old.cc
  hipCGGridGroupType_old.cc
  hipCGTiledPartitionType_old.cc
  hipCGThreadBlockTileTypeShfl_old.cc
  hipLaunchCooperativeKernel_old.cc
  hipLaunchCooperativeKernelMultiDevice_old.cc
  hipCGThreadBlockType.cc
  hipCGThreadBlockTypeViaBaseType.cc
  hipCGThreadBlockTypeViaPublicApi.cc
  hipCGMultiGridGroupType.cc
  hipCGMultiGridGroupTypeViaBaseType.cc
  hipCGMultiGridGroupTypeViaPublicApi.cc
  grid_group.cc
  coalesced_groups_shfl_down.cc
  coalesced_groups_shfl_up.cc
  hipCGTiledPartition.cc
  hipCGCoalescedGroups.cc
)
if(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(multi_grid_group.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(hipCGMultiGridGroupType_old.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(hipLaunchCooperativeKernelMultiDevice_old.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS "-rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
else()
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
endif()
