// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"

#include "core/session/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/customregistry.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/graph/schema_registry.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test_utils.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

// Foo kernel which is doing Add
template <typename T>
class FooKernel : public OpKernel {
 public:
  FooKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);

    auto X_Data = X->Data<T>();
    auto W_Data = W->Data<T>();

    auto shape = X->Shape().GetDims();

    auto* Y = context->Output(0, shape);
    auto* Y_Data = Y->MutableData<T>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i] + W_Data[i];
    }

    return Status::OK();
  }
};

ONNX_NAMESPACE::OpSchema GetFooSchema() {
  ONNX_NAMESPACE::OpSchema schema("Foo", "unknown", 0);
  schema.SetDomain("test");
  schema.Input(0,
               "A",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "B",
      "Second operand. With broadcasting can be of smaller size than A. "
      "If broadcasting is disabled it should be of the same size.",
      "T");
  schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(1);
  return schema;
}

KernelDefBuilder FooKernelDef() {
  KernelDefBuilder def;
  def.SetName("Foo")
      .SetDomain("test")
      .SinceVersion(1)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

Status CreateFooKernel(FuncManager&, const OpKernelInfo& kernel_info, std::unique_ptr<OpKernel>& out) {
  out = std::make_unique<FooKernel<float>>(kernel_info);
  return Status::OK();
}

// kernel with optional outputs
KernelDefBuilder OptionalKernelDef() {
  KernelDefBuilder def;
  def.SetName("OptionalOp")
      .SetDomain("test")
      .SinceVersion(1)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

ONNX_NAMESPACE::OpSchema GetOptionalOpSchema() {
  ONNX_NAMESPACE::OpSchema schema("OptionalOp", "unknown", 0);
  schema.SetDomain("test");
  schema.Input(0,
               "X",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "W",
      "Second operand. If provided, add it to the output",
      "T",
      OpSchema::Optional);
  schema.Output(0, "Y", "Result, has same dimensions and type as A", "T");
  schema.Output(1, "Y2", "Result, has same dimensions and type as A", "T", OpSchema::Optional);
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(1);
  return schema;
}

template <typename T>
class OptionalOpKernel : public OpKernel {
 public:
  OptionalOpKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);

    auto* X_Data = X->Data<T>();
    auto shape = X->Shape().GetDims();
    auto* Y = context->Output(0, shape);
    auto* Y_Data = Y->MutableData<T>();
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i];
    }

    auto* Y2 = context->Output(1, shape);
    // Y2 is used or not
    if (Y2) {
      auto Y2_Data = Y2->MutableData<T>();
      for (size_t i = 0; i < size; i++) {
        Y2_Data[i] = X_Data[i];
      }
    }

    // W is used or not
    if (W) {
      auto* W_Data = W->Data<T>();
      for (size_t i = 0; i < size; i++) {
        Y_Data[i] += W_Data[i];
      }
      if (Y2) {
        auto* Y2_Data = Y2->MutableData<T>();
        for (size_t i = 0; i < size; i++) {
          Y2_Data[i] += W_Data[i];
        }
      }
    }

    return Status::OK();
  }
};

Status CreateOptionalOpKernel(FuncManager&, const OpKernelInfo& kernel_info, std::unique_ptr<OpKernel>& out) {
  out = std::make_unique<OptionalOpKernel<float>>(kernel_info);
  return Status::OK();
}

static const std::string MUL_MODEL_URI = "testdata/mul_1.onnx";
static const std::string FOO_MODEL_URI = "testdata/foo_1.onnx";
static const std::string FOO_CLIP_MODEL_URI = "testdata/foo_1_clip_11.onnx";
static const std::string OPTIONAL_MODEL1_URI = "testdata/optional_1.onnx";

void RunSession(InferenceSession& session_object,
                std::vector<int64_t>& dims_x,
                std::vector<float>& values_x,
                std::vector<int64_t>& dims_y,
                std::vector<float>& values_y) {
  // prepare inputs
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // Now run
  EXPECT_STATUS_OK(session_object.Run(RunOptions{}, feeds, output_names, &fetches));
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_shape.Size());
  ASSERT_EQ(values_y, found);
}

// This tests that a custom op can override an ONNX operator implemented by ORT.
TEST(CustomKernelTests, CustomKernelWithBuiltInSchema) {
  SessionOptions so;
  so.session_logid = "CustomKernelWithBuiltInSchema";

  // Register a custom kernel that matches the ONNX Mul but is implemented to do an Add so we can validate the
  // custom kernel overrides the ORT Mul kernel
  KernelDefBuilder def;
  def.SetName("Mul")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(7)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  EXPECT_STATUS_OK(registry->RegisterCustomKernel(def, CreateFooKernel));

  InferenceSession session_object{so, GetEnvironment()};
  EXPECT_STATUS_OK(session_object.RegisterCustomRegistry(registry));
  EXPECT_STATUS_OK(session_object.Load(MUL_MODEL_URI));
  EXPECT_STATUS_OK(session_object.Initialize());

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, dims_x, values_x, expected_dims_y, expected_values_y);
}

// Test registering a custom kernel with custom schema
TEST(CustomKernelTests, CustomKernelWithCustomSchema) {
  SessionOptions so;

  so.session_logid = "CustomKernelWithCustomSchema";

  // register foo schema
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas = {GetFooSchema()};
  auto def = FooKernelDef();

  EXPECT_STATUS_OK(registry->RegisterOpSet(schemas, "test", 1, 1000));
  EXPECT_STATUS_OK(registry->RegisterCustomKernel(def, CreateFooKernel));

  InferenceSession session_object{so, GetEnvironment()};
  EXPECT_STATUS_OK(session_object.RegisterCustomRegistry(registry));
  EXPECT_STATUS_OK(session_object.Load(FOO_MODEL_URI));
  EXPECT_STATUS_OK(session_object.Initialize());

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomKernelWithOptionalOutput) {
  SessionOptions so;
  so.session_logid = "CustomKernelWithOptionalOutput";

  // register optional schema
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas = {GetOptionalOpSchema()};
  auto def = OptionalKernelDef();

  EXPECT_STATUS_OK(registry->RegisterOpSet(schemas, "test", 1, 1000));
  EXPECT_STATUS_OK(registry->RegisterCustomKernel(def, CreateOptionalOpKernel));

  InferenceSession session_object{so, GetEnvironment()};
  EXPECT_STATUS_OK(session_object.RegisterCustomRegistry(registry));
  EXPECT_STATUS_OK(session_object.Load(OPTIONAL_MODEL1_URI));
  EXPECT_STATUS_OK(session_object.Initialize());

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be equal result.
  std::vector<float> expected_values_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Now run
  RunSession(session_object, dims_x, values_x, expected_dims_y, expected_values_y);
}

// Regression test for OnnxRuntimeOpSchemaRegistry::GetSchemaAndHistory needing to reset `version` before
// falling through to the ONNX schema lookup.
//
// If there is a custom registry that matches the ONNX domain but not the current op, we fall though but need to
// use the original opset version and ignore any version values found in the custom registry.
//
// If we regress we will match Clip(1) which only had one input. The model uses Clip(11) and has two inputs. The ONNX
// checker will fail if this happens.
TEST(CustomKernelTests, CustomOnnxKernelSchemaLookup) {
  SessionOptions so;
  so.session_logid = "CustomOnnxKernelSchemaLookup";

  auto schema = GetFooSchema();
  auto def = FooKernelDef();
  schema.SetDomain(onnxruntime::kOnnxDomain);
  def.SetDomain(onnxruntime::kOnnxDomain);

  std::vector<OpSchema> schemas = {schema};
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  EXPECT_STATUS_OK(registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 1, 1000));
  EXPECT_STATUS_OK(registry->RegisterCustomKernel(def, CreateFooKernel));

  InferenceSession session_object{so, GetEnvironment()};
  EXPECT_STATUS_OK(session_object.RegisterCustomRegistry(registry));
  EXPECT_STATUS_OK(session_object.Load(FOO_CLIP_MODEL_URI));
  EXPECT_STATUS_OK(session_object.Initialize());
}
}  // namespace test
}  // namespace onnxruntime
