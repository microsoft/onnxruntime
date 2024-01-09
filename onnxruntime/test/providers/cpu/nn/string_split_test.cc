#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(StringSplit, BasicSplitTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {3}, {"hello world", "hello", "world"});
  test.AddAttribute<std::string>("delimiter", " ");
  test.AddOutput<std::string>("Y", {3, 2}, {"hello", "world", "hello", "", "world", ""});
  test.AddOutput<int64_t>("Z", {3}, {2, 1, 1});
  test.Run();
}

TEST(StringSplit, MaxSplitTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {2, 2}, {"eggs;milk;chesse", "pepper;salt", "chicken;fish;pork", "spinach"});
  test.AddAttribute<std::string>("delimiter", ";");
  test.AddAttribute<int64_t>("maxsplit", 1);
  test.AddOutput<std::string>("Y", {2, 2, 2},
                              {"eggs", "milk;chesse", "pepper", "salt", "chicken", "fish;pork", "spinach", ""});
  test.AddOutput<int64_t>("Z", {2, 2}, {2, 2, 2, 1});
  test.Run();
}

TEST(StringSplit, EmptyStringDelimiterTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {1, 4}, {"hello world", "hello  world", " hello world", "hello world  "});
  test.AddAttribute<std::string>("delimiter", "");
  test.AddOutput<std::string>("Y", {1, 4, 2}, {"hello", "world", "hello", "world", "hello", "world", "hello", "world"});
  test.AddOutput<int64_t>("Z", {1, 4}, {2, 2, 2, 2});
  test.Run();
}

TEST(StringSplit, SubsequentWhitespaceDefaultTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {1, 4}, {"hello world", "hello  world", "   hello world", "hello world  "});
  test.AddOutput<std::string>("Y", {1, 4, 2}, {"hello", "world", "hello", "world", "hello", "world", "hello", "world"});
  test.AddOutput<int64_t>("Z", {1, 4}, {2, 2, 2, 2});
  test.Run();
}

TEST(StringSplit, SubsequentWhitespaceWithLimitTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {1, 4},
                             {"lorem  ipsum doler", " Open Neural Network Exchange (ONNX)", "onnx", "ONNX runtime "});
  test.AddAttribute<int64_t>("maxsplit", 1);
  test.AddOutput<std::string>(
      "Y", {1, 4, 2},
      {"lorem", "ipsum doler", "Open", "Neural Network Exchange (ONNX)", "onnx", "", "ONNX", "runtime"});
  test.AddOutput<int64_t>("Z", {1, 4}, {2, 2, 1, 2});
  test.Run();
}

TEST(StringSplit, SingleTokenTest) {
  OpTester test("StringSplit", 20);
  test.AddAttribute<std::string>("delimiter", "*");
  test.AddInput<std::string>("X", {1, 1, 1}, {"lorem"});
  test.AddOutput<std::string>("Y", {1, 1, 1, 1}, {"lorem"});
  test.AddOutput<int64_t>("Z", {1, 1, 1}, {1});
  test.Run();
}

TEST(StringSplit, SingleTokenWhitespaceTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {1, 1, 1}, {"lorem"});
  test.AddOutput<std::string>("Y", {1, 1, 1, 1}, {"lorem"});
  test.AddOutput<int64_t>("Z", {1, 1, 1}, {1});
  test.Run();
}

TEST(StringSplit, EdgeWhitespaceTest) {
  OpTester test("StringSplit", 20);
  test.AddInput<std::string>("X", {1, 1, 1}, {"         lorem "});
  test.AddOutput<std::string>("Y", {1, 1, 1, 1}, {"lorem"});
  test.AddOutput<int64_t>("Z", {1, 1, 1}, {1});
  test.Run();
}

TEST(StringSplit, EmptyInputTest) {
  OpTester test("StringSplit", 20);
  test.AddAttribute<std::string>("delimiter", "*");
  test.AddInput<std::string>("X", {1, 3, 1}, {"", "+", "*"});
  test.AddOutput<std::string>("Y", {1, 3, 1, 2}, {"", "", "+", "", "", ""});
  test.AddOutput<int64_t>("Z", {1, 3, 1}, {0, 1, 2});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
