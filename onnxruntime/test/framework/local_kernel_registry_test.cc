// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include "core/graph/schema_registry.h"
#include "core/framework/customregistry.h"
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
  schema.SinceVersion(7);
  return schema;
}

//For test purpose, we register this Foo kernel to Mul op.
//Once the custom schema is ready, should update this.
KernelDefBuilder FooKernelDef(const char* schema_name) {
  KernelDefBuilder def;
  def.SetName(schema_name)
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(7)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

OpKernel* CreateFooKernel(const OpKernelInfo& kernel_info) {
  return new FooKernel<float>(kernel_info);
}

// kernel with optional outputs
KernelDefBuilder OptionalKernelDef() {
  KernelDefBuilder def;
  def.SetName("OptionalOp")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(6)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

ONNX_NAMESPACE::OpSchema GetOptionalOpSchema() {
  ONNX_NAMESPACE::OpSchema schema("OptionalOp", "unknown", 0);
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
  schema.SinceVersion(6);
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
    auto& shape = X->Shape().GetDims();
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

    //W is used or not
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

OpKernel* CreateOptionalOpKernel(const OpKernelInfo& kernel_info) {
  return new OptionalOpKernel<float>(kernel_info);
}

static const std::string MUL_MODEL_URI = "testdata/mul_1.onnx";
static const std::string FOO_MODEL_URI = "testdata/foo_1.onnx";
static const std::string FOO_TRUNCATE_MODEL_URI = "testdata/foo_2.onnx";

static const std::string OPTIONAL_MODEL1_URI = "testdata/optional_1.onnx";

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                std::vector<int64_t>& dims_x,
                std::vector<float>& values_x,
                std::vector<int64_t>& dims_y,
                std::vector<float>& values_y) {
  // prepare inputs
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape), *reinterpret_cast<const std::vector<int64_t>*>(&rtensor.Shape()));
  const std::vector<float> found(rtensor.template Data<float>(), rtensor.template Data<float>() + expected_shape.Size());
  ASSERT_EQ(values_y, found);
}

TEST(CustomKernelTests, CustomKernelWithBuildInSchema) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  // Register a foo kernel which is doing Add, but bind to Mul.
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  auto def = FooKernelDef("Mul");

  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateFooKernel).IsOK());

  EXPECT_TRUE(session_object.Load(MUL_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomKernelWithCustomSchema) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  //register foo schema
  auto foo_schema = GetFooSchema();
  std::vector<OpSchema> schemas = {foo_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 5, 7).IsOK());
  auto def = FooKernelDef("Foo");
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateFooKernel).IsOK());

  EXPECT_TRUE(session_object.Load(FOO_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}

TEST(CustomKernelTests, CustomKernelWithOptionalOutput) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  //reigster optional schema
  auto optional_schema = GetOptionalOpSchema();
  std::vector<OpSchema> schemas = {optional_schema};

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 5, 7).IsOK());
  auto def = OptionalKernelDef();
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateOptionalOpKernel).IsOK());

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  EXPECT_TRUE(session_object.Load(OPTIONAL_MODEL1_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be equal result.
  std::vector<float> expected_values_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
}  // namespace test
}  // namespace onnxruntime
