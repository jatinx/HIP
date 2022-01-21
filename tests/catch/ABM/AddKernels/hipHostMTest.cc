#include <hip_test_common.hh>
#include <iostream>

TEST_CASE("ABM_hipMallocManaged_Negative", "") {
  void* ptr;

  HIP_CHECK(hipMallocManaged(nullptr, 0, 0));
  HIP_CHECK(hipMallocManaged(&ptr, 0, 0));
  HIP_CHECK(hipMallocManaged(&ptr, 10, 0xFFFF));
}