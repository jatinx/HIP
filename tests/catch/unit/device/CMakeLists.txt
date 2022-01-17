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

# TODO make this a function
add_library(DeviceTestR EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
target_compile_definitions(DeviceTestR PUBLIC HIP_TEST_REPORT_MODE=1)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET DeviceTestR PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(DeviceTestR PUBLIC -std=c++17)
endif()