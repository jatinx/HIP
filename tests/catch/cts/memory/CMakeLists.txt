set(TEST_SRC malloc.cc)

hip_add_exe_to_target(NAME CTSMemory
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)