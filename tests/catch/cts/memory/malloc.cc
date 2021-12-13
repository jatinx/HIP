#include <hip_test_common.hh>

TEMPLATE_TEST_CASE("cts_hipMalloc_types", "", char, short, int, long, long long, float, double) {
  TestType* ptr{nullptr};
  HIP_CHECK(hipMalloc(&ptr, sizeof(TestType)));
  REQUIRE(ptr != nullptr);
  HIP_CHECK(hipFree(ptr));
}

TEST_CASE("cts_hipMalloc_sizes") {
  char* ptr{nullptr};
  auto size = GENERATE(1, 100);
  HIP_CHECK(hipMalloc(&ptr, sizeof(size)));
  REQUIRE(ptr != nullptr);
  HIP_CHECK(hipFree(ptr));
}