# Common Tests - Test independent of all platforms
set(TEST_SRC
    add.cc
)

hip_add_exe_to_target(NAME ABMAddKernels
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_executable(ABMAddKernelsR EXCLUDE_FROM_ALL ${TEST_SRC} $<TARGET_OBJECTS:Main_Object>)
add_dependencies(build_report ABMAddKernelsR)
target_link_libraries(ABMAddKernelsR stdc++fs) 
target_compile_definitions(ABMAddKernelsR PUBLIC HIP_TEST_REPORT_MODE=1)
