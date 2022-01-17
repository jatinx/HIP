#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_EXTERNAL_INTERFACES
#include <hip_test_common.hh>
#include <iostream>

namespace helper {
static bool fTestGroupFlag = true;
static bool fTestCaseFlag = true;
static bool fAssertionFlag = true;

std::string getQuotedString(std::string s) {
  std::string res{"\""};
  res += s;
  res += "\"";
  return res;
}

std::string boolToString(bool res) { return res ? "true" : "false"; }
std::string jsonStart() { return std::string("{\n"); }
std::string jsonEnd() { return std::string("\n}"); }

std::string arrayStart() { return std::string("[\n"); }
std::string arrayEnd() { return std::string("\n]"); }

std::string comma() { return std::string(",\n"); }
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

    f << helper::jsonStart() << helper::getQuotedString("TestRunName") << ":"
      << helper::getQuotedString(_testRunInfo.name) << helper::comma()
      << helper::getQuotedString("TestGroups") << ":" << helper::arrayStart();
  }

  virtual void testGroupStarting(Catch::GroupInfo const& _groupInfo) override {
    StreamingReporterBase::testGroupStarting(_groupInfo);
    if (!helper::fTestGroupFlag)
      f << helper::comma();
    else
      helper::fTestGroupFlag = false;
    
    f << helper::jsonStart() << helper::getQuotedString("TestGroupName") << ":"
      << helper::getQuotedString(_groupInfo.name) << ",\n"
      << helper::getQuotedString("TestCases") << ":" << helper::arrayStart();
  }

  virtual void testCaseStarting(Catch::TestCaseInfo const& _testInfo) override {
    StreamingReporterBase::testCaseStarting(_testInfo);
    if (!helper::fTestCaseFlag)
      f << helper::comma();
    else
      helper::fTestGroupFlag = false;

    f << helper::jsonStart() << helper::getQuotedString("TestCase") << ":"
      << helper::getQuotedString(_testInfo.name) << helper::comma()
      << helper::getQuotedString("Assertions") << ":" << helper::arrayStart();
  }

  virtual void assertionStarting(Catch::AssertionInfo const& assertionInfo) override {
    if (!helper::fAssertionFlag)
      f << helper::comma();
    else
      helper::fAssertionFlag = false;
    f << helper::jsonStart() << helper::getQuotedString("Name") << ":"
      << helper::getQuotedString(std::string(assertionInfo.macroName)) << helper::comma()
      << helper::getQuotedString("FileName") << ":"
      << helper::getQuotedString(std::string(assertionInfo.lineInfo.file)) << helper::comma()
      << helper::getQuotedString("Expression") << ":"
      << helper::getQuotedString(std::string(assertionInfo.capturedExpression)) << helper::comma();
  }

  virtual bool assertionEnded(Catch::AssertionStats const& assertionStats) override {
    f << helper::getQuotedString("Result") << ":"
      << helper::getQuotedString(helper::boolToString(assertionStats.assertionResult.succeeded()));


    bool isFirstm = true;
    for (const auto& i : assertionStats.infoMessages) {
      if (!isFirstm)
        f << helper::comma();
      else {
        f << helper::comma() << helper::getQuotedString("Messages") << ":" << helper::arrayStart();
      }
      f << helper::getQuotedString(i.message);
      isFirstm = false;
    }
    if (!isFirstm) f << helper::arrayEnd();
    f << helper::jsonEnd();
    return true;
  }

  virtual void testCaseEnded(Catch::TestCaseStats const& testCaseStats) override {
    StreamingReporterBase::testCaseEnded(testCaseStats);
    helper::fAssertionFlag = true;
    f << helper::arrayEnd() << helper::jsonEnd();
  }

  virtual void testGroupEnded(Catch::TestGroupStats const& testGroupStats) override {
    StreamingReporterBase::testGroupEnded(testGroupStats);
    helper::fTestCaseFlag = true;
    f << helper::arrayEnd() << helper::jsonEnd();
  }

  virtual void testRunEnded(Catch::TestRunStats const& testRunStats) override {
    StreamingReporterBase::testRunEnded(testRunStats);
    helper::fTestGroupFlag = true;
    f << helper::arrayEnd() << helper::jsonEnd();
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
