#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const std::vector<int64_t>& dims, const std::vector<std::string>& input1,
                    const std::vector<std::string>& input2, const std::vector<std::string>& output) {
  OpTester test("StringConcat", 20, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("X", dims, input1);
  test.AddInput<std::string>("Y", dims, input2);
  test.AddOutput<std::string>("Z", dims, output);
  test.Run();
}

TEST(StringConcat, BasicConcatenation) {
  RunTest({1, 2}, {"Hello", "World"}, {"Hello", "World"}, {"HelloHello", "WorldWorld"});
}

TEST(StringConcat, TwoDimensionalConcatenation) {
  RunTest({2, 2}, {"Hello", "World", "ONNX", "onnxruntime"}, {"Hello", "World", "ONNX", "onnxruntime"},
          {"HelloHello", "WorldWorld", "ONNXONNX", "onnxruntimeonnxruntime"});
}

TEST(StringConcat, LeftToRightBroadcastingConcatenation) {
  OpTester test("StringConcat", 20, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("X", {2, 2}, {"Hello", "World", "ONNX", "onnxruntime"});
  test.AddInput<std::string>("Y", {1}, {"!"});
  test.AddOutput<std::string>("Z", {2, 2}, {"Hello!", "World!", "ONNX!", "onnxruntime!"});
  test.Run();
}

TEST(StringConcat, RightToLeftBroadcastingConcatenation) {
  OpTester test("StringConcat", 20, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("X", {1}, {"!"});
  test.AddInput<std::string>("Y", {2, 2}, {"Hello", "World", "ONNX", "onnxruntime"});
  test.AddOutput<std::string>("Z", {2, 2}, {"!Hello", "!World", "!ONNX", "!onnxruntime"});
  test.Run();
}

TEST(StringConcat, BidirectionalBroadcastingConcatenation) {
  OpTester test("StringConcat", 20, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("X", {2, 1, 3}, {"a", "b", "c", "d", "e", "f"});
  test.AddInput<std::string>("Y", {1, 4, 3}, {"a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m"});
  test.AddOutput<std::string>("Z", {2, 4, 3}, {
    "aa", "aa", "aa", "aa", "aa", "aa",
    "aa", "aa", "aa", "aa", "aa", "aa",
    "aa", "aa", "aa", "aa", "aa", "aa",
    "aa", "aa", "aa", "aa", "aa", "aa",
  });
}


}  // namespace test
}  // namespace onnxruntime
