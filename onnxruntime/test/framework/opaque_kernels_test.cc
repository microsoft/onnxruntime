// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "asserts.h"
#include "test_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

const char* experimental_using_opaque = R"DOC(
The operator constructs an instance of sparse one dimensional tensor
represented by a SparseTensorSample type. It uses 3 supplied inputs each
in a form of a single dimensional tensor.
)DOC";

namespace onnxruntime {
// We will use this class to implement Sparse Tensor and
// register it as an Opaque type emulating some experimental type

/**
 * @brief This class implements a SparseTensor as an example
 *        of using custom experimental type outside of ONNXRuntime.
 *
 * @details The class captures the 3 necessary elements of a Sparse Tensor
 *          values - a vector of non-zero sparse tensor values
 *          indices - a vector of indices of non zero values
 *          shape   - a scalar tensor that indicates the size of a single dimension
 *                   It is assumed that all of the values for the tensors are int64
 *          we use tensor datatypes as effective memory managers.
 */

// This type is a result of the construct_sparse OpKernel.
class SparseTensorSample final {
 public:
  SparseTensorSample() = default;
  ~SparseTensorSample() = default;

  SparseTensorSample(const SparseTensorSample&) = default;
  SparseTensorSample& operator=(const SparseTensorSample&) = default;
  SparseTensorSample(SparseTensorSample&&) = default;
  SparseTensorSample& operator=(SparseTensorSample&&) = default;

  const std::vector<int64_t>& Values() const {
    return values_;
  }

  const std::vector<int64_t>& Indicies() const {
    return indicies_;
  }

  int64_t Size() const {
    return size_;
  }

  std::vector<int64_t>& Values() {
    return values_;
  }

  std::vector<int64_t>& Indicies() {
    return indicies_;
  }

  int64_t& Size() {
    return size_;
  }

 private:
  std::vector<int64_t> values_;
  std::vector<int64_t> indicies_;
  int64_t size_;  // The value of a single dimension
};

// We will then register this class as an Opaque type as if created and used by
// a 3rd party for experiments.
extern const char kTestDomain[] = "ai.onnx";
extern const char kSparseTensorName[] = "SparseTensorSample";

ORT_REGISTER_OPAQUE_TYPE(SparseTensorSample, kTestDomain, kSparseTensorName);

class OpaqueTypeTests : public testing::Test {
 public:
  static void SetUpTestCase() {
    MLDataType mltype = DataTypeImpl::GetType<SparseTensorSample>();
    DataTypeImpl::RegisterDataType(mltype);
  }
};

/**
 *  @brief This class represents an operator kernel which takes as input 3 tensors
 *
 * The OpKernel takes 3 tensors as input named as follows:
 * - sparse_values - Tensor
 * - sparse_indicies - Tensor
 * - sparse_shape - Tensor
 *
 *  Output - TestSparseTensorType - Opaque type
 */
class ConstructSparseTensor final : public OpKernel {
 public:
  ConstructSparseTensor(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    ORT_ENFORCE(ctx->InputCount() == 3, "Expecting 3 inputs");

    const Tensor& values_tensor = *ctx->Input<Tensor>(0);
    const Tensor& indicies_tensor = *ctx->Input<Tensor>(1);
    const Tensor& shape_tensor = *ctx->Input<Tensor>(2);

    // Shapes of values and indicies should be the same since they refer to the same
    // values
    const TensorShape& val_shape = values_tensor.Shape();
    const TensorShape& ind_shape = indicies_tensor.Shape();
    ORT_ENFORCE(val_shape.NumDimensions() == 1, "Expecting vectors");
    ORT_ENFORCE(val_shape.NumDimensions() == ind_shape.NumDimensions());

    // Copy data. With some effort we could hold shallow copies of the input Tensors
    // but I will leave this for now.
    SparseTensorSample* output_sparse_tensor = ctx->Output<SparseTensorSample>(0);
    ORT_ENFORCE(output_sparse_tensor != nullptr);
    output_sparse_tensor->Values().assign(values_tensor.Data<int64_t>(),
                                          values_tensor.Data<int64_t>() + val_shape[0]);
    output_sparse_tensor->Indicies().assign(indicies_tensor.Data<int64_t>(),
                                            indicies_tensor.Data<int64_t>() + ind_shape[0]);
    output_sparse_tensor->Size() = *shape_tensor.Data<int64_t>();

    return Status::OK();
  }
};

/**
 *  @brief This class represents an operator kernel that fetches and returns
 *         sparse tensor shape from an Opaque type
 *
 *
 *  Output - Scalar Tensor<int64_t>
 */
class FetchSparseTensorShape final : public OpKernel {
 public:
  FetchSparseTensorShape(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    ORT_ENFORCE(ctx->InputCount() == 1, "Expecting a single SparseTensorSample input");
    const SparseTensorSample* sparse_input = ctx->Input<SparseTensorSample>(0);
    // Always a single dimension of 1 bc we are storing a single number
    const int64_t dims[1] = {1};
    TensorShape output_shape(dims, 1);
    Tensor* sparse_shape = ctx->Output(0, output_shape);
    int64_t* shape_data = sparse_shape->MutableData<int64_t>();
    ORT_ENFORCE(shape_data != nullptr);
    *shape_data = sparse_input->Size();

    return Status::OK();
  }
};

namespace test {

KernelDefBuilder ConstructSparseTensorDef() {
  KernelDefBuilder def;
  def.SetName("ConstructSparseTensor")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(8)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T1",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("T2",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("T3",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("T",
                      DataTypeImpl::GetType<SparseTensorSample>());
  return def;
}

KernelDefBuilder ConstructFetchSparseShape() {
  KernelDefBuilder def;
  def.SetName("FetchSparseTensorShape")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(8)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T1",
                      DataTypeImpl::GetType<SparseTensorSample>())
      .TypeConstraint("T",
                      DataTypeImpl::GetTensorType<int64_t>());
  return def;
}

ONNX_NAMESPACE::OpSchema GetConstructSparseTensorSchema() {
  ONNX_NAMESPACE::OpSchema schema("ConstructSparseTensor", __FILE__, __LINE__);
  schema.SetDoc(experimental_using_opaque)
      .SetDomain(onnxruntime::kMLDomain)
      .Input(
          0,
          "sparse_values",
          "Single dimensional Tensor that holds all non-zero values",
          "T1",
          OpSchema::Single)
      .Input(
          1,
          "sparse_indicies",
          "Single dimensional tensor that holds indicies of non-zero values",
          "T2",
          OpSchema::Single)
      .Input(
          2,
          "sparse_shape",
          "Single dimensional tensor that holds sparse tensor shape",
          "T3",
          OpSchema::Single)
      .Output(
          0,
          "sparse_rep",
          "SparseTensorSample opaque object",
          "T",
          OpSchema::Single)
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Only int64 is allowed")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Only int64 is allowed")
      .TypeConstraint(
          "T3",
          {"tensor(int64)"},
          "Only int64 is allowed")
      .TypeConstraint(
          "T",
          {"opaque(ai.onnx,SparseTensorSample)"},
          "Opaque object");
  schema.SinceVersion(8);
  return schema;
}

ONNX_NAMESPACE::OpSchema GetFetchSparseShapeSchema() {
  ONNX_NAMESPACE::OpSchema schema("FetchSparseTensorShape", __FILE__, __LINE__);
  schema.SetDoc(experimental_using_opaque)
      .SetDomain(onnxruntime::kMLDomain)
      .Input(
          0,
          "sparse_rep",
          "Opaque SparseTensorSample",
          "T1",
          OpSchema::Single)
      .Output(
          0,
          "sparse_tensor_shape",
          "Single dimensional tensor that holds sparse tensor shape",
          "T",
          OpSchema::Single)
      .TypeConstraint(
          "T1",
          {"opaque(ai.onnx,SparseTensorSample)"},
          "Only int64 is allowed")
      .TypeConstraint(
          "T",
          {"tensor(int64)"},
          "Only int64 is allowed");
  schema.SinceVersion(8);
  return schema;
}

TEST_F(OpaqueTypeTests, RunModel) {
  SessionOptions so;
  so.session_logid = "SparseTensorTest";
  so.session_log_verbosity_level = 1;

  // Both the session and the model need custom registries
  // so we construct it here before the model
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterCustomRegistry(registry));

  auto ops_schema = GetConstructSparseTensorSchema();
  auto shape_schema = GetFetchSparseShapeSchema();
  std::vector<OpSchema> schemas = {ops_schema, shape_schema};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 8, 9));
  // Register our kernels here
  auto ctor_def = ConstructSparseTensorDef();
  ASSERT_STATUS_OK(registry->RegisterCustomKernel(
      ctor_def,
      [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
        out = std::make_unique<ConstructSparseTensor>(info);
        return Status::OK();
      }));
  auto shape_def = ConstructFetchSparseShape();
  ASSERT_STATUS_OK(registry->RegisterCustomKernel(
      shape_def,
      [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
        out = std::make_unique<FetchSparseTensorShape>(info);
        return Status::OK();
      }));

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_ = {registry->GetOpschemaRegistry()};
  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMLDomain, 8}};

  Model model("SparseTensorTest", false, ModelMetaData(), PathString(), custom_schema_registries_, domain_to_version,
              {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  TypeProto input_tensor_proto(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());

  {
    // Sparse tensor will contain total 5 elements but only 2 of them a non-zero
    TypeProto input_values(input_tensor_proto);
    input_values.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_values_arg = graph.GetOrCreateNodeArg("sparse_values", &input_values);
    inputs.push_back(&sparse_values_arg);

    TypeProto input_indicies(input_tensor_proto);
    input_indicies.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_indicies_arg = graph.GetOrCreateNodeArg("sparse_indicies", &input_indicies);
    inputs.push_back(&sparse_indicies_arg);

    // Shape tensor will contain only one value
    TypeProto input_shape(input_tensor_proto);
    input_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& sparse_shape_arg = graph.GetOrCreateNodeArg("sparse_shape", &input_shape);
    inputs.push_back(&sparse_shape_arg);

    // Output is our custom data type
    TypeProto output_sparse_tensor(*DataTypeImpl::GetType<SparseTensorSample>()->GetTypeProto());
    auto& output_sparse_tensor_arg = graph.GetOrCreateNodeArg("sparse_rep", &output_sparse_tensor);
    outputs.push_back(&output_sparse_tensor_arg);

    auto& node = graph.AddNode("ConstructSparseTensor", "ConstructSparseTensor", "Create a sparse tensor representation",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
  {
    // We start the input from previous node output
    inputs = std::move(outputs);
    outputs.clear();

    TypeProto output_shape(input_tensor_proto);
    output_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& output_shape_arg = graph.GetOrCreateNodeArg("sparse_tensor_shape", &output_shape);
    outputs.push_back(&output_shape_arg);
    auto& node = graph.AddNode("FetchSparseTensorShape", "FetchSparseTensorShape", "Fetch shape from sparse tensor",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  ASSERT_STATUS_OK(graph.Resolve());

  // Get a proto and load from it
  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;

  // Prepare inputs/outputs
  std::vector<int64_t> val_dims = {2};
  std::vector<int64_t> values = {1, 2};
  // prepare inputs
  OrtValue ml_values;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], val_dims, values, &ml_values);

  std::vector<int64_t> ind_dims = {2};
  std::vector<int64_t> indicies = {1, 4};
  OrtValue ml_indicies;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], ind_dims, indicies, &ml_indicies);

  std::vector<int64_t> shape_dims = {1};
  std::vector<int64_t> shape = {5};
  OrtValue ml_shape;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape_dims, shape, &ml_shape);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("sparse_values", ml_values));
  feeds.insert(std::make_pair("sparse_indicies", ml_indicies));
  feeds.insert(std::make_pair("sparse_shape", ml_shape));

  // Output
  std::vector<int64_t> output_shape_dims = {1};
  std::vector<int64_t> output_shape = {0};

  std::vector<std::string> output_names;
  output_names.push_back("sparse_tensor_shape");
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  // Should get the original shape back in the form of a tensor
  EXPECT_EQ(1u, rtensor.Shape().NumDimensions());
  EXPECT_EQ(5, *rtensor.Data<int64_t>());
}

}  // namespace test
}  // namespace onnxruntime
