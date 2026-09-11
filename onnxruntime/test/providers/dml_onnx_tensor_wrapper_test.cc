// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {

// Serializes |values| as little-endian INT64 raw bytes, then truncates the result to
// |truncateToByteCount| bytes when that is smaller than the full encoding.
std::string MakeInt64RawData(const std::vector<int64_t>& values, size_t truncateToByteCount = SIZE_MAX) {
  std::string raw;
  raw.reserve(values.size() * sizeof(int64_t));
  for (int64_t value : values) {
    for (size_t byteIndex = 0; byteIndex < sizeof(int64_t); ++byteIndex) {
      raw.push_back(static_cast<char>(static_cast<uint64_t>(value) >> (8 * byteIndex)));
    }
  }

  if (truncateToByteCount < raw.size()) {
    raw.resize(truncateToByteCount);
  }

  return raw;
}

// How the 'value' attribute of the ConstantOfShape node should be populated.
enum class ValueAttributePayload {
  FullRawData,       // 8 bytes of raw_data - well formed.
  TruncatedRawData,  // 1 byte of raw_data for a tensor that declares one INT64 element.
};

// Builds a single-node model:
//
//   ConstantOfShape(shape) -> Y      with attribute value : INT64[1]
//
// where 'shape' is a well-formed INT64[1] initializer holding the value 4, so Y is INT64[4] filled
// with the single element of the 'value' attribute.
//
// The DML EP registers ConstantOfShape with requiredConstantCpuInputs(0), and
// DmlOperatorConstantOfShape reads the 'value' attribute through
// IMLOperatorAttributes1::GetTensorAttribute, which wraps the raw AttributeProto tensor in an
// OnnxTensorWrapper.
//
// Attribute tensors are not graph initializers, so none of the framework-level guards apply to them:
// Graph::ConvertInitializersIntoOrtValues never sees them, and ONNX shape inference for
// ConstantOfShape only reads the attribute's data_type (to type the output), never its data. That
// leaves OnnxTensorWrapper as the only place the payload can be validated.
std::string BuildConstantOfShapeModel(ValueAttributePayload payload) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  ONNX_NAMESPACE::OperatorSetIdProto& opset = *model.add_opset_import();
  opset.set_domain("");
  opset.set_version(13);

  ONNX_NAMESPACE::GraphProto& graph = *model.mutable_graph();
  graph.set_name("dml_onnx_tensor_wrapper_test");

  ONNX_NAMESPACE::ValueInfoProto& output = *graph.add_output();
  output.set_name("Y");
  output.mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  ONNX_NAMESPACE::TensorProto& shape = *graph.add_initializer();
  shape.set_name("shape");
  shape.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape.add_dims(1);
  shape.set_raw_data(MakeInt64RawData({4}));

  ONNX_NAMESPACE::NodeProto& node = *graph.add_node();
  node.set_name("constant_of_shape0");
  node.set_op_type("ConstantOfShape");
  node.add_input("shape");
  node.add_output("Y");

  ONNX_NAMESPACE::AttributeProto& valueAttribute = *node.add_attribute();
  valueAttribute.set_name("value");
  valueAttribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);

  // Every variant declares a single INT64 element, i.e. a shape that implies 8 bytes of payload.
  ONNX_NAMESPACE::TensorProto& valueTensor = *valueAttribute.mutable_t();
  valueTensor.set_name("value");
  valueTensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  valueTensor.add_dims(1);

  switch (payload) {
    case ValueAttributePayload::FullRawData:
      valueTensor.set_raw_data(MakeInt64RawData({42}));
      break;
    case ValueAttributePayload::TruncatedRawData:
      valueTensor.set_raw_data(MakeInt64RawData({42}, /*truncateToByteCount*/ 1));
      break;
  }

  std::string serialized;
  EXPECT_TRUE(model.SerializeToString(&serialized));
  return serialized;
}

// Loads |serializedModel| into a session that has only the DML EP registered, and returns the status
// of Initialize(). |skipped| is set when no DirectML device is available.
//
// Constant folding is disabled so the ConstantOfShape node survives graph optimization and is handed
// to the DML EP. Without this, the CPU EP would evaluate the node during Level1 optimization and the
// DML code under test would never run.
Status InitializeWithDmlEp(const std::string& serializedModel,
                           bool& skipped,
                           std::unique_ptr<InferenceSession>* outSession = nullptr) {
  skipped = false;

  std::unique_ptr<IExecutionProvider> dmlEp = DefaultDmlExecutionProvider();
  if (dmlEp == nullptr) {
    skipped = true;
    return Status::OK();
  }

  SessionOptions sessionOptions;
  sessionOptions.session_logid = "DmlOnnxTensorWrapperTest";
  ORT_RETURN_IF_ERROR(sessionOptions.config_options.AddConfigEntry(
      kOrtSessionOptionsDisableSpecifiedOptimizers, "ConstantFolding"));

  auto session = std::make_unique<InferenceSession>(sessionOptions, GetEnvironment());
  ORT_RETURN_IF_ERROR(session->RegisterExecutionProvider(std::move(dmlEp)));
  ORT_RETURN_IF_ERROR(session->Load(serializedModel.data(), static_cast<int>(serializedModel.size())));

  Status status = session->Initialize();
  if (outSession != nullptr) {
    *outSession = std::move(session);
  }

  return status;
}

}  // namespace

// Regression test for an out-of-bounds read in OnnxTensorWrapper.
//
// OnnxTensorWrapper reports the tensor shape straight from TensorProto::dims but sizes its data
// buffer from whichever payload the proto actually carries, and never cross-checks the two.
// DmlOperatorConstantOfShape then reads sizeof(element type) bytes out of that buffer, so a 'value'
// attribute that declares one INT64 element while storing a single byte produces a 7-byte heap
// over-read whose contents are copied into the DML fill pattern - and therefore into the model
// output.
TEST(DmlOnnxTensorWrapperTest, TruncatedRawDataAttributeTensorIsRejected) {
  const std::string model = BuildConstantOfShapeModel(ValueAttributePayload::TruncatedRawData);

  bool skipped = false;
  const Status status = InitializeWithDmlEp(model, skipped);
  if (skipped) {
    GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
  }

  ASSERT_FALSE(status.IsOK()) << "Session initialization should reject a tensor whose declared shape "
                                 "exceeds the data it carries.";

  // The detailed message raised inside the constructor does not survive: OnnxTensorWrapper is
  // constructed behind the IMLOperatorAttributes1 COM boundary, whose ORT_CATCH_RETURN reduces the
  // exception to a bare HRESULT before it is rethrown. 0x80070057 is E_INVALIDARG, so this asserts the
  // model was rejected as malformed rather than failing for an unrelated reason.
  EXPECT_NE(status.ErrorMessage().find("80070057"), std::string::npos)
      << "Unexpected failure reason: " << status.ErrorMessage();
}

// The constructor's typed-field branch has the same defect in principle: UnpackTensor sizes the buffer
// from the repeated field's element count rather than from dims. It is not covered by a test because no
// malformed proto can actually reach it through this operator - every way of constructing one is
// rejected earlier:
//   - omitting the value field entirely fails the ONNX checker ("should contain one and only one value
//     field");
//   - storing the payload in a field that does not match the declared type fails the ONNX checker
//     ("values of data_type '7' should be stored in field 'int64_data' instead of 'int32_data'");
//   - declaring more elements than the field holds is caught by DmlOperatorConstantOfShape's own
//     elementCount == 1 assertion before it reads any bytes.
// The constructor-level check added alongside this test guards that branch regardless, so a future
// consumer without an equivalent assertion cannot reintroduce the read.

// Positive control: a well-formed attribute tensor must still initialize and produce the right fill
// pattern, so the validation above cannot regress valid models.
TEST(DmlOnnxTensorWrapperTest, WellFormedAttributeTensorIsAccepted) {
  const std::string model = BuildConstantOfShapeModel(ValueAttributePayload::FullRawData);

  bool skipped = false;
  std::unique_ptr<InferenceSession> session;
  const Status status = InitializeWithDmlEp(model, skipped, &session);
  if (skipped) {
    GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
  }

  ASSERT_STATUS_OK(status);

  RunOptions runOptions;
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session->Run(runOptions, NameMLValMap{}, std::vector<std::string>{"Y"}, &fetches));

  ASSERT_EQ(fetches.size(), 1u);
  const Tensor& result = fetches[0].Get<Tensor>();
  ASSERT_EQ(result.Shape(), TensorShape({4}));

  for (int64_t element : result.DataAsSpan<int64_t>()) {
    EXPECT_EQ(element, 42);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_DML
