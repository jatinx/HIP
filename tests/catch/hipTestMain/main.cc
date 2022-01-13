#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_EXTERNAL_INTERFACES
#include <hip_test_common.hh>
#include <iostream>

namespace helper {
std::string getQuotedString(std::string s) {
  std::string res{"\""};
  res += s;
  res += "\"";
  return res;
}

std::string jsonStart() { return std::string("{\n"); }
std::string jsonEnd() { return std::string("\n}"); }

std::string arrayStart() { return std::string("[\n"); }
std::string arrayEnd() { return std::string("\n]"); }
}  // namespace helper

class HIPReporter : public Catch::StreamingReporterBase<HIPReporter> {
  FileStreamer f;

 public:
  using StreamingReporterBase::StreamingReporterBase;

  static std::string getDescription() {
    return "Reporter for logging output of HIP APIs as json file";
  }

  virtual void testRunStarting(Catch::TestRunInfo const& _testRunInfo) override {
    StreamingReporterBase::testRunStarting(_testRunInfo);
    std::cout << "TestRunStart : " << _testRunInfo.name << std::endl;
  }

  virtual void testGroupStarting(Catch::GroupInfo const& _groupInfo) override {
    StreamingReporterBase::testGroupStarting(_groupInfo);
    std::cout << "TestGroupStart : " << _groupInfo.name << std::endl;
  }

  virtual void testCaseStarting(Catch::TestCaseInfo const& _testInfo) override {
    StreamingReporterBase::testCaseStarting(_testInfo);
    std::cout << "TestCaseStart : " << _testInfo.name << std::endl;
  }

  virtual void assertionStarting(Catch::AssertionInfo const& assertionInfo) override {
    std::cout << "AssertionStart : Macro Name : " << assertionInfo.macroName
              << " Line No : " << assertionInfo.lineInfo
              << " Expression : " << assertionInfo.capturedExpression
              << " Result Disposition: " << assertionInfo.resultDisposition << std::endl;
  }

  virtual bool assertionEnded(Catch::AssertionStats const& assertionStats) override {
    std::cout << "AssertionEnd : Result : " << assertionStats.assertionResult.succeeded()
              << std::endl;
    return true;
  }

  virtual void testCaseEnded(Catch::TestCaseStats const& testCaseStats) override {
    StreamingReporterBase::testCaseEnded(testCaseStats);
    std::cout << "TestCaseEnd : " << testCaseStats.testInfo.name << std::endl;
  }

  virtual void testGroupEnded(Catch::TestGroupStats const& testGroupStats) override {
    StreamingReporterBase::testGroupEnded(testGroupStats);
    std::cout << "TestGroupEnd : " << testGroupStats.groupInfo.name << std::endl;
  }

  virtual void testRunEnded(Catch::TestRunStats const& testRunStats) override {
    StreamingReporterBase::testRunEnded(testRunStats);
    std::cout << "TestRunEnd : " << testRunStats.runInfo.name << std::endl;
  }
};

CATCH_REGISTER_REPORTER("hip", HIPReporter)

int main(int argc, char** argv) {
  auto& context = TestContext::get(argc, argv);
  if (context.skipTest()) {
    // CTest uses this regex to figure out if the test has been skipped
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    return 0;
  }
  return Catch::Session().run(argc, argv);
}
