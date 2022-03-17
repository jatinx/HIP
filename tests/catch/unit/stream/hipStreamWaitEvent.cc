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


#include <hip_test_common.hh>

TEST_CASE("Unit_hipStreamWaitEvent_Negative") {
  enum class StreamTestType { NullStream = 0, StreamPerThread, CreatedStream };

  auto streamType = GENERATE(StreamTestType::NullStream, StreamTestType::StreamPerThread,
                             StreamTestType::CreatedStream);

  hipStream_t stream{nullptr};
  hipEvent_t event{nullptr};

  if (streamType == StreamTestType::StreamPerThread) {
    stream = hipStreamPerThread;
  } else if (streamType == StreamTestType::CreatedStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  HIP_CHECK(hipEventCreate(&event));

  REQUIRE((stream != nullptr) != (streamType == StreamTestType::NullStream));
  REQUIRE(event != nullptr);

  SECTION("Invalid Event") {
    INFO("Running against Invalid Event");
    HIP_CHECK_ERROR(hipStreamWaitEvent(stream, nullptr, 0), hipErrorInvalidResourceHandle);
  }

  SECTION("Invalid Flags") {
    INFO("Running against Invalid Flags");
    constexpr unsigned flag = ~0u;
    REQUIRE(flag != 0);
    HIP_CHECK_ERROR(hipStreamWaitEvent(stream, event, flag), hipErrorInvalidValue);
  }

  HIP_CHECK(hipEventDestroy(event));

  if (streamType == StreamTestType::CreatedStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

// Kernel that runs a lambda which is passed on to it, with any arguments, thats passed along side
// lambda.
template <typename F, typename... Args> __global__ void launchFunctor(F f, Args... args) {
  f(args...);
}

TEST_CASE("Unit_hipStreamWaitEvent_DifferentStreams") {
  hipStream_t blockedStreamA{nullptr}, streamBlockedOnStreamA{nullptr}, unblockingStream{nullptr};
  hipEvent_t waitEvent{nullptr};

  HIP_CHECK(hipStreamCreate(&blockedStreamA));
  HIP_CHECK(hipStreamCreate(&streamBlockedOnStreamA));
  HIP_CHECK(hipStreamCreate(&unblockingStream));
  HIP_CHECK(hipEventCreate(&waitEvent));

  REQUIRE(blockedStreamA != nullptr);
  REQUIRE(streamBlockedOnStreamA != nullptr);
  REQUIRE(waitEvent != nullptr);

  auto setValLambda = [] __device__(int* ptr, int val) { *ptr = val; };
  auto waitLambda = [] __device__(int* sema) {
    while (atomicCAS(sema, 1, 0) == 0)  //  CAS 1 with 0, basically unset it
      ;
  };

  int *d_a{nullptr}, *waitData{nullptr};
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));
  HIP_CHECK(hipMallocManaged(&waitData, sizeof(int)));

  HIP_CHECK(hipMemset(d_a, 0, sizeof(int)));
  HIP_CHECK(hipMemset(waitData, 0, sizeof(int)));

  launchFunctor<<<1, 1, 0, blockedStreamA>>>(setValLambda, d_a, 1);  // d_a = 1
  launchFunctor<<<1, 1, 0, blockedStreamA>>>(waitLambda,
                                             waitData);  // Wait on waitData to have a value val
  HIP_CHECK(hipEventRecord(waitEvent, blockedStreamA));

  // Make sure stream is waiting for data to be set
  HIP_CHECK_ERROR(hipEventQuery(waitEvent), hipErrorNotReady);

  HIP_CHECK(hipStreamWaitEvent(streamBlockedOnStreamA, waitEvent, 0));

  launchFunctor<<<1, 1, 0, streamBlockedOnStreamA>>>(setValLambda, d_a, -1);  // d_a = -1

  // Make sure stream is waiting for event on blockedStreamA
  HIP_CHECK_ERROR(hipStreamQuery(streamBlockedOnStreamA), hipErrorNotReady);

  // Release the wait Data
  *waitData = 1;
  // launchFunctor<<<1, 1, 0, unblockingStream>>>(setValLambda, waitData, 1);  // unblock streamA
  // HIP_CHECK(hipStreamSynchronize(unblockingStream));

  HIP_CHECK(hipStreamSynchronize(blockedStreamA));
  HIP_CHECK(hipStreamSynchronize(streamBlockedOnStreamA));

  // Check that both streams have finished
  HIP_CHECK(hipStreamQuery(blockedStreamA));
  HIP_CHECK(hipStreamQuery(streamBlockedOnStreamA));

  // Check if kernel on streamBlockedOnStreamA has done its job
  int result{0};
  HIP_CHECK(hipMemcpyAsync(&result, d_a, sizeof(int), hipMemcpyDeviceToHost, unblockingStream));
  REQUIRE(result == -1);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(waitData));
  HIP_CHECK(hipStreamDestroy(blockedStreamA));
  HIP_CHECK(hipStreamDestroy(streamBlockedOnStreamA));
  HIP_CHECK(hipStreamDestroy(unblockingStream));
  HIP_CHECK(hipEventDestroy(waitEvent));
}
