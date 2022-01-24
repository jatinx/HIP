#include <hip_test_common.hh>
#include <iostream>

template <typename Lambda> __global__ void kernel(float* a, Lambda func) { func(a); }

TEST_CASE("ABM_sin_range", "") {
  float* dval;
  HIP_CHECK(hipMalloc(&dval, sizeof(float)));

  auto lambda = __device__[](float* a) { *a = asinf(*a); };

  for (int i = 100; i <= 1000; i++) {
    float val = i / 100.0f;
    HIP_CHECK(hipMemcpy(dval, &val, sizeof(float), hipMemcpyHostToDevice));
    auto k = kernel<lambda>;
    k<<<1, 1>>>(dval);
    HIP_CHECK(hipMemcpy(&val, dval, sizeof(float), hipMemcpyDeviceToHost));
    HIP_LOG_RESULT(val, "SinResult")
  }
  HIP_CHECK(hipFree(dval));
}