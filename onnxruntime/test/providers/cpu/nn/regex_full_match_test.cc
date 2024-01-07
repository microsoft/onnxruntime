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

TEST(RegexFullMatch, MultibyteMatch) {
  RunTest({1, 2}, {"ä", "a"}, "ä", {true, false});
  RunTest({1,}, {"une cédille like in Besançon"}, R"(.*cédille.*)", {true,});
  RunTest({1,}, {"une cédille like in Besançon"}, R"(.*cedille.*)", {false,});
  RunTest({1,}, {"Mit freundlichen Grüßen"}, R"(.*Grüßen$)", {true,});
  RunTest({1,}, {"Mit freundlichen Grüßen"}, R"(.*Grußen$)", {false,});
  RunTest({3,}, {"HПонедельник", "Понедельник", "недельник"}, R"(^Понед.*)", {false, true, false,});
  RunTest({3,}, {"thank you", "どうもありがとうございます", "こんにちは世界"}, R"(^こんにちは世界.*)", {false, false, true,});
  RunTest({3,}, {"नमस्ते, आपसे मिलकर अच्छा लगा", "नमस्ते", "स्वागत एवं नमस्ते"}, R"(.+नमस्ते$)", {false, false, true,});
  RunTest({3,}, {"你好，你好吗?", "你好呀", "你好呀!"}, R"(^你好.*\?$)", {true, false, false,});
}

TEST(RegexFullMatch, InvalidPattern) {
  OpTester test("RegexFullMatch", 20, kOnnxDomain);
  test.AddAttribute("pattern", R"([a-z)");
  test.AddInput<std::string>("Input", {1,}, {"abcdef",});
  test.AddOutput<bool>("Output", {1,}, {false,});
  test.Run(BaseTester::ExpectResult::kExpectFailure, "Invalid regex pattern: [a-z");
}

}  // namespace test
}  // namespace onnxruntime
