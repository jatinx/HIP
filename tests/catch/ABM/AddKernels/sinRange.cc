#include <hip_test_common.hh>
#include <iostream>

template <typename Lambda> __global__ void kernel(float* a, Lambda func) { func(a); }

TEST_CASE("ABM_sin_range", "") {
  float* dval;
  HIP_CHECK(hipMalloc(&dval, sizeof(float)));

  auto lambda = [] __device__ (float* a) { *a = asinf(*a); };

  for (int i = 100; i <= 300; i++) {
    float val = i / 100.;
    float res = 0.0f;
    HIP_CHECK(hipMemcpy(dval, &val, sizeof(float), hipMemcpyHostToDevice));
    kernel<<<1, 1>>>(dval, lambda);
    HIP_CHECK(hipMemcpy(&res, dval, sizeof(float), hipMemcpyDeviceToHost));
    HIP_LOG_RESULT(val, std::to_string(res) + "SinResult")
  }
  HIP_CHECK(hipFree(dval));
}


TEST_CASE("ABM_cos_range", "") {
  float* dval;
  HIP_CHECK(hipMalloc(&dval, sizeof(float)));

  auto lambda = [] __device__ (float* a) { *a = acosf(*a); };

  for (int i = 100; i <= 300; i++) {
    float val = i / 100.;
    float res = 0.0f;
    HIP_CHECK(hipMemcpy(dval, &val, sizeof(float), hipMemcpyHostToDevice));
    kernel<<<1, 1>>>(dval, lambda);
    HIP_CHECK(hipMemcpy(&res, dval, sizeof(float), hipMemcpyDeviceToHost));
    HIP_LOG_RESULT(val, std::to_string(res) + "CosResult")
  }
  HIP_CHECK(hipFree(dval));
}