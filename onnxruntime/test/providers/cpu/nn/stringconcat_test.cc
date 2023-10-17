#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const std::vector<int64_t>& dims, const std::vector<std::string>& input1, const std::vector<std::string>& input2, const std::vector<std::string>& output) {
    std::cout << "Running test\n";
    OpTester test("StringConcat", 20, onnxruntime::kOnnxDomain);
    test.AddInput<std::string>("X", dims, input1);
    std::cout << "add input1" << "\n";
    test.AddInput<std::string>("Y", dims, input2);
    test.AddOutput<std::string>("Z", dims, output);
    test.Run();
}

TEST(StringConcat, BasicConcatenation) {
    RunTest({1, 2}, {"Hello", "World"}, {"Hello", "World"}, {"HelloHello", "WorldWorld"});
}

TEST(StringConcat, TwoDimensionalConcatenation) {
    RunTest({2, 2}, {"Hello", "World", "ONNX", "onnxruntime"}, {"Hello", "World", "ONNX", "onnxruntime"}, {"HelloHello", "WorldWorld", "ONNXONNX", "onnxruntimeonnxruntime"});
}


} // namespace test
} // namespace onnxruntime
