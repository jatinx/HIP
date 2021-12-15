#include <hip_test_common.hh>

TEST_CASE("ctsxl_malloc", "") {
  constexpr long long count = 1000000;
  for (long long i = 0; i < count; i++) {
    int* ptr{nullptr};
    INFO("Running iter:" << i << "/" << count);
    HIP_CHECK(hipMalloc(&ptr, sizeof(int) * 100));
    HIP_CHECK(hipFree(ptr));
  }
}