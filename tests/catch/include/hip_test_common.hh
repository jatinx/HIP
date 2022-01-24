/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include "hip_test_context.hh"
#include <catch.hpp>
#include <stdlib.h>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

#if defined(HIP_TEST_REPORT_MODE)

#define HIP_CHECK(hipApi)                                                                          \
  {                                                                                                \
    auto __hip_api_res = (hipApi);                                                                 \
    INFO("FILE::" << __FILE__);                                                                    \
    INFO("HIPAPI::" << #hipApi);                                                                   \
    INFO("LINENO::" << __LINE__);                                                                  \
    INFO("HIPRES::" << hipGetErrorName(__hip_api_res));                                            \
    REQUIRE(true);                                                                                 \
  }

#define HIPRTC_CHECK(hiprtcApi)                                                                    \
  {                                                                                                \
    auto __hiprtc_api_res = (hiprtcApi);                                                           \
    INFO("FILE::" << __FILE__);                                                                    \
    INFO("HIPAPI::" << #hiprtcApi);                                                                \
    INFO("LINENO::" << __LINE__);                                                                  \
    INFO("HIPRES::" << hiprtcGetErrorName(__hip_api_res));                                         \
    REQUIRE(true);                                                                                 \
  }

#define HIP_NCHECK(hipApi) HIP_CHECK(hipApi)

#define HIP_LOG_RESULT(result, name)                                                               \
  {                                                                                                \
    INFO("FILE::" << __FILE__);                                                                    \
    INFO("LINENO::" << __LINE__);                                                                  \
    INFO("RESULTNAME::" << name);                                                                  \
    INFO("RESULT::" << result);                                                                    \
    REQUIRE(true);
}

#else

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      INFO("Error: " << hipGetErrorString(localError) << " Code: " << localError << " Str: "       \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

#define HIPRTC_CHECK(error)                                                                        \
  {                                                                                                \
    auto localError = error;                                                                       \
    if (localError != HIPRTC_SUCCESS) {                                                            \
      INFO("Error: " << hiprtcGetErrorString(localError) << " Code: " << localError << " Str: "    \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

#define HIP_NCHECK(error)                                                                          \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if (localError == hipSuccess) {                                                                \
      INFO("Error: " << hipGetErrorString(localError) << " Code: " << localError << " Str: "       \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

#define HIP_LOG_RESULT(result, name)

#endif

// Although its assert, it will be evaluated at runtime
#define HIP_ASSERT(x)                                                                              \
  { REQUIRE((x)); }

#ifdef __cplusplus
#include <iostream>
#include <iomanip>
#include <chrono>
#endif

#define HIPCHECK(error) HIP_CHECK(error)

#define HIPASSERT(condition) HIP_ASSERT(condition)


// Utility Functions
namespace HipTest {
static inline int getDeviceCount() {
  int dev = 0;
  HIP_CHECK(hipGetDeviceCount(&dev));
  return dev;
}

// Returns the current system time in microseconds
static inline long long get_time() {
  return std::chrono::high_resolution_clock::now().time_since_epoch() /
      std::chrono::microseconds(1);
}

static inline double elapsed_time(long long startTimeUs, long long stopTimeUs) {
  return ((double)(stopTimeUs - startTimeUs)) / ((double)(1000));
}

static inline unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N) {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  unsigned blocks = props.multiProcessorCount * blocksPerCU;
  if (blocks * threadsPerBlock > N) {
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  }

  return blocks;
}

static inline int RAND_R(unsigned* rand_seed) {
#if defined(_WIN32) || defined(_WIN64)
  srand(*rand_seed);
  return rand();
#else
  return rand_r(rand_seed);
#endif
}
}  // namespace HipTest
