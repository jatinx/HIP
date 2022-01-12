#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_EXTERNAL_INTERFACES
#include <hip_test_common.hh>
#include <iostream>

//#if defined(HIP_TEST_REPORT_MODE)

class PartialReporter : public Catch::StreamingReporterBase<PartialReporter> {
 public:
  using StreamingReporterBase::StreamingReporterBase;

  static std::string getDescription() {
    return "Reporter for testing TestCasePartialStarting/Ended events";
  }

  void testCasePartialStarting(Catch::TestCaseInfo const& testInfo, uint64_t partNumber) {
    std::cout << "TestCaseStartingPartial: " << testInfo.name << '#' << partNumber << '\n';
  }

  void testCasePartialEnded(Catch::TestCaseStats const& testCaseStats,
                            uint64_t partNumber) {
    std::cout << "TestCasePartialEnded: " << testCaseStats.testInfo.name << '#' << partNumber
              << '\n';
  }
  virtual void assertionStarting( Catch::AssertionInfo const& assertionInfo ) { std::cout << assertionInfo.macroName << " - " << assertionInfo.capturedExpression << std::endl; }
        virtual bool assertionEnded( Catch::AssertionStats const& assertionStats ) { return true; }
};


CATCH_REGISTER_REPORTER("partial", PartialReporter)

//#endif

int main(int argc, char** argv) {
  auto& context = TestContext::get(argc, argv);
  if (context.skipTest()) {
    // CTest uses this regex to figure out if the test has been skipped
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    return 0;
  }
  return Catch::Session().run(argc, argv);
}
