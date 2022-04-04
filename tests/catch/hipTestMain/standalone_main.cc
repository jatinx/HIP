#define CATCH_CONFIG_MAIN
#include "hip_test_common.hh"

namespace internal {
struct HCResult {
  size_t line;
  std::string file;
  hipError_t result;
  std::string call;
  HCResult(size_t l, std::string f, hipError_t r, std::string c)
      : line(l), file(f), result(r), call(c) {}
};

std::vector<HCResult> hcResults;
std::mutex resultMutex;
std::atomic<bool> hasErrorOccured{false};
}  // namespace internal
