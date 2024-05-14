#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void TestShape(const std::initializer_list<T>& data, const std::vector<int64_t>& shape) {
  OpTester test("Shape");
  test.AddInput<T>("data", shape, data);
  test.AddOutput<int64_t>("output", {static_cast<int64_t>(shape.size())}, shape);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

TEST(ShapeOpTest, ShapeTestBool) { TestShape<bool>({true, true, false, false, true, false}, {2, 3}); }
TEST(ShapeOpTest, ShapeTestFloat) { TestShape<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 6}); }
TEST(ShapeOpTest, ShapeTestDouble) { TestShape<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {6, 2}); }
TEST(ShapeOpTest, ShapeTestInt8) { TestShape<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4}); }
TEST(ShapeOpTest, ShapeTestInt16) { TestShape<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4}); }
TEST(ShapeOpTest, ShapeTestInt32) { TestShape<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3}); }
TEST(ShapeOpTest, ShapeTestInt64) { TestShape<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}); }
TEST(ShapeOpTest, ShapeTestUint8) { TestShape<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {12, 1}); }
TEST(ShapeOpTest, ShapeTestUint16) { TestShape<uint16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}); }
TEST(ShapeOpTest, ShapeTestUint32) { TestShape<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {12, 1}); }
TEST(ShapeOpTest, ShapeTestUint64) { TestShape<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}); }
TEST(ShapeOpTest, ShapeTestString) { TestShape<std::string>({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}, {1, 12}); }

TEST(ShapeOpTest, ShapeOpset15_Default) {
  OpTester test("Shape", 15);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {3}, {1, 2, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

TEST(ShapeOpTest, ShapeOpset15_StartOnly) {
  OpTester test("Shape", 15);
  test.AddAttribute<int64_t>("start", 1);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {2}, {2, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

TEST(ShapeOpTest, ShapeOpset15_EndOnly) {
  OpTester test("Shape", 15);
  test.AddAttribute<int64_t>("end", 2);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {2}, {1, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

TEST(ShapeOpTest, ShapeOpset15_StartAndEnd) {
  OpTester test("Shape", 15);
  test.AddAttribute<int64_t>("start", 1);
  test.AddAttribute<int64_t>("end", 2);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {1}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

TEST(ShapeOpTest, ShapeOpset15_StartAndEndNegative) {
  OpTester test("Shape", 15);
  test.AddAttribute<int64_t>("start", -2);
  test.AddAttribute<int64_t>("end", -1);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {1}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}
TEST(ShapeOpTest, ShapeOpset15_StartAndEndProducingEmptySlice) {
  OpTester test("Shape", 15);
  test.AddAttribute<int64_t>("start", 2);
  test.AddAttribute<int64_t>("end", 2);
  test.AddInput<int32_t>("data", {1, 2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser: unsupported data types
}

}  // namespace test
}  // namespace onnxruntime
