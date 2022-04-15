/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <atomic>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <mutex>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    INFO("Matching Error to hipSuccess or hipErrorPeerAccessAlreadyEnabled: "                      \
         << hipGetErrorString(localError) << "\n    Code: " << localError << "\n    Str: "         \
         << #error << "\n    In File: " << __FILE__ << "\n    At line: " << __LINE__);             \
    REQUIRE(((localError == hipSuccess) || (localError == hipErrorPeerAccessAlreadyEnabled)));     \
  }

// Check that an expression, errorExpr, evaluates to the expected error_t, expectedError.
#define HIP_CHECK_ERROR(errorExpr, expectedError)                                                  \
  {                                                                                                \
    hipError_t localError = errorExpr;                                                             \
    INFO("Matching Errors:"                                                                        \
         << "\n    Expected Error: " << hipGetErrorString(expectedError)                           \
         << "\n    Expected Code: " << expectedError                                               \
         << "\n    Actual Error:   " << hipGetErrorString(localError)                              \
         << "\n    Actual Code:   " << localError << "\n    Str: " << #errorExpr                   \
         << "\n    In File: " << __FILE__ << "\n    At line: " << __LINE__);                       \
    REQUIRE(localError == expectedError);                                                          \
  }

inline namespace internal {
struct HCResult {
  size_t line;
  std::string file;
  hipError_t result;
  std::string call;
  bool condition;
  HCResult(size_t l, std::string f, hipError_t r, std::string c, bool cres = true)
      : line(l), file(f), result(r), call(c), condition(cres) {}
};

static std::vector<HCResult> hcResults;  // Store results to validate at the end of threads so that
                                         // we can have proper test result count
static std::mutex resultMutex;
static std::atomic<bool> hasErrorOccured{false};  // flag to stop execution of threads if error has
                                                  // occurred in one of the threads
}  // namespace internal

// Threaded HIP_CHECKs
#define HIP_CHECK_THREAD(error)                                                                    \
  {                                                                                                \
    /*To see if error has occured in previous threads, stop execution to save time waiting for     \
     * error*/                                                                                     \
    if (hasErrorOccured.load() == true) {                                                          \
      return; /*This will only work with std::thread and not with std::async*/                     \
    }                                                                                              \
    auto localError = error;                                                                       \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      hasErrorOccured.store(true);                                                                 \
    }                                                                                              \
    internal::HCResult result(__LINE__, __FILE__, localError, #error);                             \
    { /* Scoped lock */                                                                            \
      std::unique_lock<std::mutex> lock(internal::resultMutex);                                    \
      internal::hcResults.push_back(result);                                                       \
    }                                                                                              \
  }

#define REQUIRE_THREAD(condition)                                                                  \
  {                                                                                                \
    /*To see if error has occured in previous threads, stop execution to save time waiting for     \
     * error*/                                                                                     \
    if (hasErrorOccured.load() == true) {                                                          \
      return; /*This will only work with std::thread and not with std::async*/                     \
    }                                                                                              \
    auto localResult = (condition);                                                                \
    if (!localResult) {                                                                            \
      hasErrorOccured.store(true);                                                                 \
    }                                                                                              \
    HCResult result(__LINE__, __FILE__, hipSuccess, #condition, localResult);                      \
  }

// Do not call before all threads have joined
#define HIP_CHECK_THREAD_FINALIZE()                                                                \
  {                                                                                                \
    if (hasErrorOccured.load() == true) {                                                          \
      UNSCOPED_INFO("Error has Occured");                                                          \
      hasErrorOccured.store(false);                                                                \
    }                                                                                              \
    for (const auto& i : hcResults) {                                                              \
      INFO("HIP API Result check\n    File:: "                                                     \
           << i.file << "\n    Line:: " << i.line << "\n    API:: " << i.call << "\n    Result:: " \
           << i.result << "\n    Result Str:: " << hipGetErrorString(i.result));                   \
      REQUIRE(((i.result == hipSuccess) || (i.result == hipErrorPeerAccessAlreadyEnabled)));       \
      REQUIRE(i.condition);                                                                        \
    }                                                                                              \
    hcResults.clear();                                                                             \
  }

#define HIPRTC_CHECK(error)                                                                        \
  {                                                                                                \
    auto localError = error;                                                                       \
    INFO("Matching Error to HIPRTC_SUCCESS: "                                                      \
         << hiprtcGetErrorString(localError) << "\n    Code: " << localError << "\n    Str: "      \
         << #error << "\n    In File: " << __FILE__ << "\n    At line: " << __LINE__);             \
    REQUIRE(error == HIPRTC_SUCCESS);                                                              \
  }

// Although its assert, it will be evaluated at runtime
#define HIP_ASSERT(x)                                                                              \
  { REQUIRE((x)); }

#define HIPCHECK(error)                                                                            \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FILE__, __LINE__);                                                          \
      abort();                                                                                     \
    }                                                                                              \
  }

#define HIPASSERT(condition)                                                                       \
  if (!(condition)) {                                                                              \
    printf("assertion %s at %s:%d \n", #condition, __FILE__, __LINE__);                            \
    abort();                                                                                       \
  }

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

/**
 * Causes the test to stop and be skipped at runtime.
 * reason: Message describing the reason the test has been skipped.
 */
inline void HIP_SKIP_TEST(char const* const reason) noexcept {
  std::cout << "Skipping test. Reason: " << reason << '\n' << "HIP_SKIP_THIS_TEST" << std::endl;
}
}  // namespace HipTest
