#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const std::initializer_list<int64_t>& dims, const std::initializer_list<std::string>& input, const std::string& pattern, const std::initializer_list<bool>& output) {
  OpTester test("RegexFullMatch", 20, kOnnxDomain);
  test.AddAttribute("pattern", pattern);
  test.AddInput<std::string>("Input", dims, input);
  test.AddOutput<bool>("Output", dims, output);
  test.Run();
}

TEST(RegexFullMatch, WebsiteMatch) {
  RunTest({3, 1}, {"www.google.com", "www.facebook.com", "www.bbc.co.uk"}, R"(www\.[\w.-]+\.\bcom\b)", {true, true, false});
}

TEST(RegexFullMatch, EmailMatch) {
  RunTest({2, 2}, {"account@gmail.com", "account@hotmail.com", "not email", "account@yahoo.com"}, R"((\W|^)[\w.\-]{0,25}@(yahoo|gmail)\.com(\W|$))", {true, false, false, true});
}

}  // namespace test
}  // namespace onnxruntime
