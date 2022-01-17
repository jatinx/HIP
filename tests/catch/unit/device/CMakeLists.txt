# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipChooseDevice.cc
    hipDeviceComputeCapability.cc
    hipDeviceGetByPCIBusId.cc
    hipDeviceGetLimit.cc
    hipDeviceGetName.cc
    hipDeviceGetPCIBusId.cc
    hipDeviceSetGetCacheConfig.cc
    hipDeviceSynchronize.cc
    hipDeviceTotalMem.cc
    hipGetDeviceAttribute.cc
    hipGetDeviceCount.cc
    hipGetDeviceProperties.cc
    hipRuntimeGetVersion.cc
    hipSetDeviceFlags.cc
    hipSetGetDevice.cc
)

hip_add_exe_to_target(NAME DeviceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

# TODO make this a cmake function
add_executable(DeviceTestR EXCLUDE_FROM_ALL ${TEST_SRC} $<TARGET_OBJECTS:Main_Object>)
add_dependencies(build_report DeviceTestR)
target_link_libraries(DeviceTestR stdc++fs) 
target_compile_definitions(DeviceTestR PUBLIC HIP_TEST_REPORT_MODE=1)