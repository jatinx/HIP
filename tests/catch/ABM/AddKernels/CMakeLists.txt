# Common Tests - Test independent of all platforms
set(TEST_SRC
    add.cc
    hipHostMTest.cc
    sinRange.cc
)

hip_add_exe_to_target(NAME ABMAddKernels
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

# TODO make this a function
add_library(ABMAddKernelsR EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
target_compile_definitions(ABMAddKernelsR PUBLIC HIP_TEST_REPORT_MODE=1)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET ABMAddKernelsR PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(ABMAddKernelsR PUBLIC -std=c++17)
endif()