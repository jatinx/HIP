#define CATCH_CONFIG_MAIN
#include "hip_test_common.hh"

namespace internal {
std::vector<HCResult> hcResults;
std::mutex resultMutex;
std::atomic<bool> hasErrorOccured{false};
}  // namespace internal
