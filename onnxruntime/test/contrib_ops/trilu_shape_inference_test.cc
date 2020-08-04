#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"
#include "shape_inference_test_helper.h"

namespace onnxruntime {
namespace test {

TEST(ShapeInferenceTests, tri_upper) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_zero_dim) {
  std::vector<int64_t> shape = {4, 7, 0};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_4d) {
  std::vector<int64_t> shape = {2, 3, 7, 11};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_lower_4d) {
  std::vector<int64_t> shape = {2, 3, 7, 11};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(0);  // lower
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_lower_zero_dim) {
  std::vector<int64_t> shape = {4, 7, 0};
  ONNX_NAMESPACE::ValueInfoProto input;
  createValueInfo(input, "X", shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(0);  // lower
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo(output, "Y", shape);

  testShapeInference("Trilu", inputs, attributes, output);
}

}  // namespace test
}  // namespace onnxruntime
