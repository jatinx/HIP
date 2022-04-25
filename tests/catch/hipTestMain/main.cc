#define CATCH_CONFIG_RUNNER
#include <hip_test_common.hh>
#include <iostream>

int main(int argc, char** argv) {
  // Register function to run after main
  if (0 != std::atexit(HipTest::hip_test_at_exit_handler)) {
    std::cerr << "at_exit_handler resigtration failed" << std::endl;
    return 1;
  }

  auto& context = TestContext::get(argc, argv);
  if (context.skipTest()) {
    // CTest uses this regex to figure out if the test has been skipped
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    return 0;
  }
  return Catch::Session().run(argc, argv);
}
