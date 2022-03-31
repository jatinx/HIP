#include <hip_test_common.hh>

#include <thread>
#include <vector>

TEST_CASE("LetsGetThatBread?") {
  auto thread = []() {  // the thread
    constexpr size_t size = 100;
    HIP_CHECK(hipMalloc(nullptr, size));
  };
  constexpr size_t num = 10;  // number of threads

  std::vector<std::thread> threadPool;
  threadPool.reserve(num);
  for (size_t i = 0; i < num; i++) {
    threadPool.emplace_back(std::thread(thread));
  }

  for (size_t i = 0; i < num; i++) {
    threadPool[i].join();
  }
}