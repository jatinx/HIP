#include <hip_test_common.hh>
#include <iostream>

template <typename T> __global__ void add(T* a, T* b, T* c, size_t size) {
  size_t i = threadIdx.x;
  if (i < size) c[i] = a[i] + b[i];
}

TEMPLATE_TEST_CASE("ABM_AddKernel_MultiTypeMultiSize", "", int, long, float, long long, double) {
  auto size = GENERATE(as<size_t>{}, 100, 500, 1000);
  TestType *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc(&d_a, sizeof(TestType) * size));

  HIP_CHECK(hipMalloc(&d_b, sizeof(TestType) * size));
  HIP_CHECK(hipMalloc(&d_c, sizeof(TestType) * size));

  std::vector<TestType> a, b, c;
  for (size_t i = 0; i < size; i++) {
    a.push_back(i + 1);
    b.push_back(i + 1);
    c.push_back(2 * (i + 1));
  }

  HIP_CHECK(hipMemcpy(d_a, a.data(), sizeof(TestType) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, b.data(), sizeof(TestType) * size, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(add<TestType>, 1, size, 0, 0, d_a, d_b, d_c, size);

  HIP_CHECK(hipMemcpy(a.data(), d_c, sizeof(TestType) * size, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  for (unsigned long i = 0; i < size; i++) {
    INFO("Iter: " << i)
//    REQUIRE(a[i] == c[i]);
  }
}
