#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void TestShape(const std::initializer_list<T>& data,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& expected_shape,
               std::vector<int64_t>* p_perm = nullptr) {
  OpTester test("TransposeOfShape", 1, onnxruntime::kMSDomain);
  test.AddInput<T>("data", shape, data);
  test.AddOutput<int64_t>("output", {static_cast<int64_t>(shape.size())}, expected_shape);
  if (nullptr != p_perm) {
    test.AddAttribute("perm", *p_perm);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: unsupported data types
}

TEST(TransposeOfShapeOpTest, ShapeTestBool) { TestShape<bool>({true, true, false, false, true, false}, {2, 3}, {3, 2}); }
TEST(TransposeOfShapeOpTest, ShapeTestFloat) { TestShape<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 6}, {6, 2}); }
TEST(TransposeOfShapeOpTest, ShapeTestDouble) { TestShape<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {6, 2}, {2, 6}); }
TEST(TransposeOfShapeOpTest, ShapeTestInt8) { TestShape<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4}, {4, 3}); }
TEST(TransposeOfShapeOpTest, ShapeTestInt16) { TestShape<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4}, {4, 3}); }
TEST(TransposeOfShapeOpTest, ShapeTestInt32) { TestShape<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3}, {3, 4}); }
TEST(TransposeOfShapeOpTest, ShapeTestInt64) { TestShape<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}, {12, 1}); }
TEST(TransposeOfShapeOpTest, ShapeTestUint8) { TestShape<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {12, 1}, {1, 12}); }
TEST(TransposeOfShapeOpTest, ShapeTestUint16) { TestShape<uint16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}, {12, 1}); }
TEST(TransposeOfShapeOpTest, ShapeTestUint32) { TestShape<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {12, 1}, {1, 12}); }
TEST(TransposeOfShapeOpTest, ShapeTestUint64) { TestShape<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 12}, {12, 1}); }
TEST(TransposeOfShapeOpTest, ShapeTestString) { TestShape<std::string>({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}, {1, 12}, {12, 1}); }

TEST(TransposeOfShapeOpTest, ShapeTestFloatPerm) {
  std::vector<int64_t> perm = {1, 2, 0};
  TestShape<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 6}, {2, 6, 1}, &perm);
}

}  // namespace test
}  // namespace onnxruntime
