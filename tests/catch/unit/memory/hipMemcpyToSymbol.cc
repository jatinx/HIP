/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

__device__ int devSymbol[10];

/* Test verifies hipMemcpyToSymbol API Negative scenarios.
 */
TEST_CASE("Unit_hipMemcpyToSymbol_Negative") {
  SECTION("Invalid Src Ptr") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyToSymbol(nullptr, &result, sizeof(int), 0, hipMemcpyHostToDevice),
                    hipErrorInvalidSymbol);
  }

  SECTION("Invalid Dst Ptr") {
    HIP_CHECK_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), nullptr, sizeof(int), 0, hipMemcpyHostToDevice),
        hipErrorInvalidValue);
  }

  SECTION("Invalid Size") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), &result, sizeof(int) * 10, 0,
                                      hipMemcpyHostToDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Offset") {
    int result{0};
    HIP_CHECK_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), &result, sizeof(int), 3, hipMemcpyHostToDevice),
        hipErrorInvalidValue);
  }

  SECTION("Invalid Direction") {
    int result{0};
    HIP_CHECK_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), &result, sizeof(int), 0, hipMemcpyDeviceToHost),
        hipErrorInvalidMemcpyDirection);
  }
}

TEST_CASE("Unit_hipMemcpyToFromSymbol_SimpleUsecase") {
  SECTION("Singular Value") {
    int set = 42;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), &set, sizeof(int)));
    int result{0};
    HIP_CHECK(hipMemcpyFromSymbol(&result, HIP_SYMBOL(devSymbol), sizeof(int)));
    REQUIRE(result == set);
  }

  SECTION("Array Values") {
    constexpr size_t size = 10;
    int set[size] = {4, 2, 4, 2, 4, 2, 4, 2, 4, 2};
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set, sizeof(int) * size));
    int result[size] = {0};
    HIP_CHECK(hipMemcpyFromSymbol(&result, HIP_SYMBOL(devSymbol), sizeof(int) * size));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }

  SECTION("Offset'ed Values") {
    constexpr size_t size = 10;
    constexpr size_t offset = 5 * sizeof(int);
    int set[size] = {9, 9, 9, 9, 9, 2, 4, 2, 4, 2};
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set, offset));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set + 5, 0, offset));
    int result[size] = {0};
    HIP_CHECK(hipMemcpyFromSymbol(result, HIP_SYMBOL(devSymbol), sizeof(int) * size));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }
}
