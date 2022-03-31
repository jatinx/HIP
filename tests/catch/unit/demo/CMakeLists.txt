set(TEST_SRC demo.cc)

hip_add_exe_to_target(NAME DemoTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
