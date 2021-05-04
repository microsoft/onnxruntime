// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"

#include "core/graph/onnx_protobuf.h"

#include "core/graph/constants.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/tensorprotoutils.h"

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test_utils.h"
#include "file_util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

// This file contains sample implementations of several ops with sparse-tensor inputs/outputs.
// Each op is implemented as a struct with the following signature:
// struct SparseOp {
//    static std::string OpName();
//    static ONNX_NAMESPACE::OpSchema OpSchema();
//    class OpKernelImpl final : public OpKernel {...};
//    static KernelDefBuilder KernelDef();
// }

// op SparseFromCOO
struct SparseFromCOO {
  static std::string OpName() { return "SparseFromCOO"; };

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc(R"DOC(
This operator constructs a sparse tensor from three tensors that provide a COO
(coordinate) representation.
)DOC")
        .Input(
            0,
            "values",
            "A 1-dimensional tensor of shape [NNZ] that holds all non-zero values",
            "T1",
            OpSchema::Single)
        .Input(
            1,
            "indices",
            "A 2-dimensional tensor of shape [NNZ,d] that holds the (d) indices of non-zero values",
            "T2",
            OpSchema::Single)
        .Input(
            2,
            "shape",
            "A 1-dimensional tensor of shape [d] that holds the shape of the underlying dense tensor",
            "T2",
            OpSchema::Single)
        .Output(
            0,
            "sparse_rep",
            "A sparse representation of the tensor",
            "T",
            OpSchema::Single)
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Type of the values (input tensor)")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Type of index tensor and shape")
        .TypeConstraint(
            "T",
            {"sparse_tensor(int64)"},
            "Output type");
    return schema;
  }

  /**
 *  @brief An implementation of the SparseFromCOO op.
 */
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 3, "Expecting 3 inputs");

      const Tensor& values = *ctx->Input<Tensor>(0);
      const Tensor& indices = *ctx->Input<Tensor>(1);
      const Tensor& shape_tensor = *ctx->Input<Tensor>(2);

      // values and indices should be 1-dimensional tensors
      const TensorShape& val_shape = values.Shape();
      const TensorShape& ind_shape = indices.Shape();
      const TensorShape& shape_shape = shape_tensor.Shape();

      auto nnz = val_shape.Size();
      auto numdims = shape_shape.Size();

      ORT_ENFORCE(val_shape.NumDimensions() == 1, "Values must be a 1-dimensional tensor.");

      ORT_ENFORCE(ind_shape.NumDimensions() == 2, "Indices must be a 2-dimensional tensor.");
      ORT_ENFORCE(ind_shape[0] == nnz, "Indices must have [NNZ,d] shape.");
      ORT_ENFORCE(ind_shape[1] == numdims, "Indices must have [NNZ,d] shape.");

      ORT_ENFORCE(shape_shape.NumDimensions() == 1, "Shape must be a 1-dimensional tensor.");

      TensorShape shape(shape_tensor.Data<int64_t>(), shape_shape.Size());

      SparseTensor* output = ctx->Output(0, static_cast<size_t>(nnz), shape);
      ORT_ENFORCE(output != nullptr);

      memcpy(output->MutableValues().MutableData<int64_t>(), values.Data<int64_t>(), nnz * sizeof(int64_t));
      memcpy(output->MutableIndices().MutableData<int64_t>(), indices.Data<int64_t>(), nnz * numdims * sizeof(int64_t));

      return Status::OK();
    }
  };

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(SparseFromCOO::OpName())
        .TypeConstraint("values", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("indices", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("sparse_rep", DataTypeImpl::GetSparseTensorType<int64_t>());
    return def;
  }
};

// op SparseAbs
struct SparseAbs {
  static const std::string OpName() { return "SparseAbs"; };

  // The ONNX schema for SparseAbs:
  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc(R"DOC(
This operator applies the Abs op element-wise to the input sparse-tensor.
)DOC")
        .Input(
            0,
            "input",
            "Input sparse tensor",
            "T",
            OpSchema::Single)
        .Output(
            0,
            "output",
            "Output sparse tensor",
            "T",
            OpSchema::Single)
        .TypeConstraint(
            "T",
            {"sparse_tensor(int64)"},
            "Input and Output type");
    return schema;
  }

  // A kernel implementation of SparseAbs:
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 1, "Expecting 1 input");

      const SparseTensor* input = ctx->Input<SparseTensor>(0);
      auto* input_values = input->Values().Data<int64_t>();
      auto nnz = input->NumValues();
      auto& shape = input->Shape();

      // Allocate/get output-tensor:
      SparseTensor* output = ctx->Output(0, nnz, shape);

      // compute output values:
      auto* output_values = output->MutableValues().MutableData<int64_t>();
      for (size_t i = 0; i < nnz; ++i)
        output_values[i] = std::abs(input_values[i]);

      // Currently, there is no way to share the indices/shape between two sparse-tensors.
      // So, we copy indices/shape from input to output.
      // TODO: Extend allocation-planner to enable such sharing.
      const auto& input_indices = input->Indices();
      memcpy(output->MutableIndices().MutableData<int64_t>(), input_indices.Data<int64_t>(), input_indices.SizeInBytes());
      return Status::OK();
    }
  };

  // A KernelDefBuilder for SparseAbs:
  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName())
        .TypeConstraint("T", DataTypeImpl::GetSparseTensorType<int64_t>());
    return def;
  }
};

// op SparseToValues
struct SparseToValues {
  static const std::string OpName() { return "SparseToValues"; };

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Return the non-zero values in a sparse tensor (as a dense tensor).")
        .Input(
            0,
            "sparse_rep",
            "Input sparse tensor.",
            "T1",
            OpSchema::Single)
        .Output(
            0,
            "values",
            "A single dimensional tensor that holds non-zero values in the input",
            "T2",
            OpSchema::Single)
        .TypeConstraint(
            "T1",
            {"sparse_tensor(int64)"},
            "Only int64 is allowed")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Type of the values component");
    return schema;
  }

  //  A kernel implementation of SparseToValues
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 1, "Expecting a single SparseTensorSample input");
      const SparseTensor* sparse_input = ctx->Input<SparseTensor>(0);
      const auto* values = sparse_input->Values().Data<int64_t>();
      auto nnz = static_cast<int64_t>(sparse_input->NumValues());

      TensorShape output_shape{nnz};

      Tensor* output = ctx->Output(0, output_shape);
      int64_t* output_data = output->MutableData<int64_t>();
      ORT_ENFORCE(output_data != nullptr);

      memcpy(output_data, values, nnz * sizeof(int64_t));

      return Status::OK();
    }
  };

  // A KernelDefBuilder for SparseToValues
  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName())
        .TypeConstraint("sparse_rep", DataTypeImpl::GetSparseTensorType<int64_t>())
        .TypeConstraint("values", DataTypeImpl::GetTensorType<int64_t>());
    return def;
  }
};

using Action = std::function<void(CustomRegistry*)>;

class SparseTensorTests : public testing::Test {
 protected:
  InferenceSession session_object;
  std::shared_ptr<CustomRegistry> registry;
  std::unique_ptr<Model> model;

  std::vector<OpSchema> schemas;
  std::vector<Action> register_actions;
  std::vector<TypeProto> types;

 public:
  SparseTensorTests() : session_object(SessionOptions(), GetEnvironment()),
                        registry(std::make_shared<CustomRegistry>()) {
    EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  }

  template <typename Op>
  void Add() {
    auto schema = Op::OpSchema();
    schema.SetName(Op::OpName());
    schema.SetDomain(onnxruntime::kMLDomain);
    schema.SinceVersion(10);
    schemas.push_back(schema);

    Action register_kernel = [](CustomRegistry* registry2) {
      auto kernel_def_builder = Op::KernelDef();
      kernel_def_builder
          .SetDomain(onnxruntime::kMLDomain)
          .SinceVersion(10)
          .Provider(onnxruntime::kCpuExecutionProvider);
      KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) { return new typename Op::OpKernelImpl(info); };
      EXPECT_TRUE(registry2->RegisterCustomKernel(kernel_def_builder, kernel_create_fn).IsOK());
    };
    register_actions.push_back(register_kernel);
  }

  void RegisterOps() {
    EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 10, 11).IsOK());
    for (auto& registerop : register_actions)
      registerop(registry.get());
  }

  void BuildModel() {
    IOnnxRuntimeOpSchemaRegistryList custom_schema_registries = {registry->GetOpschemaRegistry()};
    model.reset(new Model("SparseTensorTest", false, ModelMetaData(), PathString(), custom_schema_registries,
                          {}, {}, DefaultLoggingManager().DefaultLogger()));
  }

  void SerializeAndLoad() {
    // Serialize model and deserialize it back
    std::string serialized_model;
    auto model_proto = model->ToProto();
    EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
    std::stringstream sstr(serialized_model);
    EXPECT_TRUE(session_object.Load(sstr).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());
  }

  NodeArg* Sparse(std::string name) {
    types.push_back(*DataTypeImpl::GetSparseTensorType<int64_t>()->GetTypeProto());
    Graph& graph = model->MainGraph();
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  NodeArg* Dense(std::string name) {
    types.push_back(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());
    Graph& graph = model->MainGraph();
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  void Node(std::string op, const std::vector<NodeArg*> inputs, const std::vector<NodeArg*> outputs) {
    Graph& graph = model->MainGraph();
    auto& node = graph.AddNode("", op, "", inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  MLValue Constant(const std::vector<int64_t>& elts) {
    const std::vector<int64_t> shape{static_cast<int64_t>(elts.size())};
    return Constant(elts, shape);
  }

  MLValue Constant(const std::vector<int64_t>& elts, const std::vector<int64_t>& shape) {
    MLValue mlvalue;
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape, elts, &mlvalue);
    return mlvalue;
  }

  NameMLValMap feeds;

  void AddInput(NodeArg* arg, const std::vector<int64_t>& value) {
    feeds[arg->Name()] = Constant(value);
  }

  void AddInput(NodeArg* arg, const std::vector<int64_t>& value, const std::vector<int64_t>& shape) {
    feeds[arg->Name()] = Constant(value, shape);
  }

  void ExpectEq(MLValue val1, MLValue val2) {
    // Restricted to case where val1 and val2 are int64_t tensors
    auto& tensor1 = val1.Get<Tensor>();
    auto& tensor2 = val2.Get<Tensor>();
    EXPECT_EQ(tensor1.Shape().Size(), tensor2.Shape().Size());
    auto* data1 = tensor1.Data<int64_t>();
    auto* data2 = tensor2.Data<int64_t>();
    for (int64_t i = 0, limit = tensor1.Shape().Size(); i < limit; ++i) {
      EXPECT_EQ(data1[i], data2[i]);
    }
  }

  void ExpectEq(MLValue val1, const std::vector<int64_t>& data2) {
    // Restricted to case where val1 is an int64_t tensor
    auto& tensor1 = val1.Get<Tensor>();
    EXPECT_EQ(static_cast<uint64_t>(tensor1.Shape().Size()), data2.size());
    auto* data1 = tensor1.Data<int64_t>();
    for (int64_t i = 0, limit = tensor1.Shape().Size(); i < limit; ++i) {
      EXPECT_EQ(data1[i], data2[i]);
    }
  }

  std::vector<std::string> output_names;
  std::vector<MLValue> expected_output;

  void ExpectOutput(NodeArg* arg, const std::vector<int64_t>& value) {
    output_names.push_back(arg->Name());
    expected_output.push_back(Constant(value));
  }

  void RunTest() {
    RunOptions run_options;
    std::vector<MLValue> fetches;

    EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());

    ASSERT_EQ(expected_output.size(), fetches.size());
    for (size_t i = 0; i < fetches.size(); ++i) {
      ExpectEq(fetches[i], expected_output[i]);
    }
  }
};

// Test ops SparseFromCOO, SparseAbs, and SparseToValues.
// Tests 1-dimensional int64 sparse tensor.
TEST_F(SparseTensorTests, Test1) {
  // Register ops
  Add<SparseFromCOO>();
  Add<SparseAbs>();
  Add<SparseToValues>();
  RegisterOps();

  // Build model/graph
  BuildModel();

  // Node: create a sparse tensor from COO components:
  // sparse1 <- SparseFromCOO(values, indices, shape)
  auto values = Dense("values");     // a dense tensor containing non-zero-values
  auto indices = Dense("indices");   // a dense tensor containing indices of non-zero-values
  auto shape = Dense("shape");       // a dense tensor representing shape
  auto sparse1 = Sparse("sparse1");  // a sparse tensor created from above

  Node(SparseFromCOO::OpName(), {values, indices, shape}, {sparse1});

  // Node:apply SparseAbs op
  // sparse2 <- SparseAbs(sparse1)
  auto sparse2 = Sparse("sparse2");
  Node(SparseAbs::OpName(), {sparse1}, {sparse2});

  // Node: Extract non-zero values from sparse2
  // output <- SparseToValues(sparse2)
  auto output = Dense("output");
  Node(SparseToValues::OpName(), {sparse2}, {output});

  // Check graph, serialize it and deserialize it back
  Graph& graph = model->MainGraph();
  EXPECT_TRUE(graph.Resolve().IsOK());
  SerializeAndLoad();

  // Run the model
  AddInput(values, {-99, 2});
  AddInput(indices, {1, 4}, {2, 1});
  AddInput(shape, {5});
  ExpectOutput(output, {99, 2});
  RunTest();
}

// Test ops SparseFromCOO, SparseAbs, and SparseToValues.
// Tests 2-dimensional int64 sparse tensor.
TEST_F(SparseTensorTests, Test2) {
  // Register ops
  Add<SparseFromCOO>();
  Add<SparseAbs>();
  Add<SparseToValues>();
  RegisterOps();

  // Build model/graph
  BuildModel();

  // Node: create a sparse tensor from COO components:
  // sparse1 <- SparseFromCOO(values, indices, shape)
  auto values = Dense("values");     // a dense tensor containing non-zero-values
  auto indices = Dense("indices");   // a dense tensor containing indices of non-zero-values
  auto shape = Dense("shape");       // a dense tensor representing shape
  auto sparse1 = Sparse("sparse1");  // a sparse tensor created from above

  Node(SparseFromCOO::OpName(), {values, indices, shape}, {sparse1});

  // Node:apply SparseAbs op
  // sparse2 <- SparseAbs(sparse1)
  auto sparse2 = Sparse("sparse2");
  Node(SparseAbs::OpName(), {sparse1}, {sparse2});

  // Node: Extract non-zero values from sparse2
  // output <- SparseToValues(sparse2)
  auto output = Dense("output");
  Node(SparseToValues::OpName(), {sparse2}, {output});

  // Check graph, serialize it and deserialize it back
  Graph& graph = model->MainGraph();
  EXPECT_TRUE(graph.Resolve().IsOK());
  SerializeAndLoad();

  // Run the model
  AddInput(values, {-99, 2});
  AddInput(indices, {1, 1, 4, 4}, {2, 2});
  AddInput(shape, {5, 5});
  ExpectOutput(output, {99, 2});
  RunTest();
}

// Code below depends on the values being size 4
template <typename T>
static std::vector<T> CreateValues() {
  return {1, 2, 3, 4};
}

/* std::string support in the future
template <>
std::vector<std::string> CreateValues<std::string>() {
  return {"one", "two", "three", "four"};
}
*/

template <>
std::vector<BFloat16> CreateValues<BFloat16>() {
  return {BFloat16(1.f), BFloat16(2.f), BFloat16(3.f), BFloat16(4.f)};
}

template <>
std::vector<MLFloat16> CreateValues<MLFloat16>() {
  return {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)};
}

template <typename T>
static void CreateTensorWithExternalData(
    TensorProto_DataType type,
    const std::vector<T>& test_data,
    std::basic_string<ORTCHAR_T>& filename,
    TensorProto& tensor_proto) {
  // Create external data
  FILE* fp;
  CreateTestFile(fp, filename);
  size_t size_in_bytes = test_data.size() * sizeof(T);
  ASSERT_EQ(size_in_bytes, fwrite(test_data.data(), 1, size_in_bytes, fp));
  ASSERT_EQ(0, fclose(fp));

  // set the tensor_proto to reference this external data
  onnx::StringStringEntryProto* location = tensor_proto.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToMBString(filename));
  tensor_proto.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  tensor_proto.set_data_type(type);
}

template <typename T>
static NodeProto CreateConstantNode(bool indices_1D,
                                    std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
                                    std::vector<T>& expected_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("dense_tensor_output");

  std::vector<T> values = CreateValues<T>();
  std::vector<int64_t> indices;
  std::vector<int64_t> shape{2, 3, 2};

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();
  attrib.set_name("sparse_value");
  attrib.set_type(AttributeProto_AttributeType_SPARSE_TENSOR);

  SparseTensorProto& stp = *attrib.mutable_sparse_tensor();
  TensorProto& indices_tp = *stp.mutable_indices();

  stp.mutable_dims()->Add(shape.cbegin(), shape.cend());
  for (auto dim : stp.dims())
    std::cout << dim;

  if (indices_1D) {
    indices = {2, 5, 6, 10};
    indices_tp.add_dims(indices.size());
  } else {
    // indices are shape {NNZ, rank} so convert flattened values of 2, 5, 6 and 10 to rank 3 values
    indices_tp.add_dims(values.size());
    indices_tp.add_dims(shape.size());
    indices = {
        0, 1, 0,
        0, 2, 1,
        1, 0, 0,
        1, 2, 0};
  }

  indices_tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices_tp.mutable_int64_data()->Add(indices.cbegin(), indices.cend());

  expected_data.resize(2 * 3 * 2);
  expected_data[2] = values[0];
  expected_data[5] = values[1];
  expected_data[6] = values[2];
  expected_data[10] = values[3];

  stp.mutable_values()->add_dims(values.size());
  inserter(values, *stp.mutable_values());

  return constant_node;
}

template <typename T>
static NodeProto CreateConstantNodeAllZeros(bool indices_1D, std::vector<T>& expected_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("dense_tensor_output");

  std::vector<int64_t> indices;
  std::vector<int64_t> shape{2, 3, 2};

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();
  attrib.set_name("sparse_value_all_zeros");
  attrib.set_type(AttributeProto_AttributeType_SPARSE_TENSOR);

  SparseTensorProto& stp = *attrib.mutable_sparse_tensor();
  TensorProto& indices_tp = *stp.mutable_indices();

  stp.mutable_dims()->Add(shape.cbegin(), shape.cend());

  if (indices_1D) {
    indices_tp.add_dims(0);
  } else {
    // indices are shape {NNZ, rank} so convert flattened values of 2, 5, 6 and 10 to rank 3 values
    indices_tp.add_dims(0);
    indices_tp.add_dims(0);
  }

  indices_tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  // Must be all zeros
  expected_data.resize(2 * 3 * 2);

  auto& mutable_values = *stp.mutable_values();
  mutable_values.set_name("all_zeros");
  mutable_values.set_data_type(utils::ToTensorProtoElementType<T>());
  mutable_values.add_dims(0);

  return constant_node;
}

template <typename T>
static void TestConversion(bool use_1D_indices,
                           std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
                           std::function<void(gsl::span<const T> expected, const TensorProto& actual)> checker) {
  std::vector<T> expected;
  auto node = CreateConstantNode<T>(use_1D_indices, inserter, expected);

  TensorProto dense;
  // Path is required for loading external data (if any)
  // When path is empty it will look for the data in current dir
  utils::ConstantNodeProtoToTensorProto(node, Path(), dense);

  gsl::span<const T> expected_span = gsl::make_span<const T>(expected.data(), expected.size());
  checker(expected_span, dense);
}

template <typename T>
static void TestConversionAllZeros(bool use_1D_indices,
                                   std::function<void(gsl::span<const T> expected, const TensorProto& actual)> checker) {
  std::vector<T> expected;
  auto node = CreateConstantNodeAllZeros<T>(use_1D_indices, expected);

  TensorProto dense;
  // Path is required for loading external data (if any)
  // When path is empty it will look for the data in current dir
  utils::ConstantNodeProtoToTensorProto(node, Path(), dense);

  gsl::span<const T> expected_span = gsl::make_span<const T>(expected.data(), expected.size());
  checker(expected_span, dense);
}

template <typename T>
static void TestConversion(
    std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
    std::function<void(gsl::span<const T> expected, const TensorProto& actual)> checker) {
  TestConversion(true, inserter, checker);
  TestConversion(false, inserter, checker);
  TestConversionAllZeros(true, checker);
  TestConversionAllZeros(false, checker);
}

template <typename T>
static void RawDataWriter(const std::vector<T>& values, TensorProto& tp, TensorProto_DataType datatype) {
  tp.set_data_type(datatype);
  tp.set_raw_data(values.data(), values.size() * sizeof(T));
}

int64_t ActualSize(const TensorProto& actual) {
  int64_t actual_size = 1;
  for (const auto dim : actual.dims()) {
    actual_size *= dim;
  }
  return actual_size;
}

template <typename T>
static void RawDataChecker(gsl::span<const T> expected, const TensorProto& actual) {
  int64_t actual_size = ActualSize(actual);

  const T* raw_data = reinterpret_cast<const T*>(actual.raw_data().data());
  auto actual_span = gsl::make_span<const T>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
}

template <>
void RawDataChecker<MLFloat16>(gsl::span<const MLFloat16> expected_bfloat, const TensorProto& actual) {
  int64_t actual_size = ActualSize(actual);

  auto expected = expected_bfloat.as_span<const uint16_t>();
  const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(actual.raw_data().data());
  auto actual_span = gsl::make_span<const uint16_t>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
}

template <>
void RawDataChecker<BFloat16>(gsl::span<const BFloat16> expected_bfloat, const TensorProto& actual) {
  int64_t actual_size = ActualSize(actual);

  auto expected = expected_bfloat.as_span<const uint16_t>();
  const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(actual.raw_data().data());
  auto actual_span = gsl::make_span<const uint16_t>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
}

TEST(SparseTensorConversionTests, TestConstantNodeConversion) {
  TestConversion<float>(
      [](const std::vector<float>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_FLOAT);
        tp.mutable_float_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<float>);

  TestConversion<double>(
      [](const std::vector<double>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_DOUBLE);
        tp.mutable_double_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<double>);

  TestConversion<BFloat16>(
      [](const std::vector<BFloat16>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_BFLOAT16);
        for (auto v : values) {
          tp.mutable_int32_data()->Add(v.val);
        }
      },
      RawDataChecker<BFloat16>);

  TestConversion<MLFloat16>(
      [](const std::vector<MLFloat16>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_FLOAT16);
        for (auto v : values) {
          tp.mutable_int32_data()->Add(v.val);
        }
      },
      RawDataChecker<MLFloat16>);

  TestConversion<int16_t>(
      [](const std::vector<int16_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT16);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int16_t>);

  TestConversion<uint16_t>(
      [](const std::vector<uint16_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_UINT16);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<uint16_t>);

  TestConversion<int32_t>(
      [](const std::vector<int32_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT32);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int32_t>);

  TestConversion<uint32_t>(
      [](const std::vector<uint32_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_UINT32);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<uint32_t>);

  // Test all zeros case
  TestConversion<int64_t>(
      [](const std::vector<int64_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT64);
        tp.mutable_int64_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int64_t>);

  TestConversion<uint64_t>(
      [](const std::vector<uint64_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_UINT64);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<uint64_t>);

  TestConversion<int8_t>(
      [](const std::vector<int8_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT8);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int8_t>);

  TestConversion<uint8_t>(
      [](const std::vector<uint8_t>& values, TensorProto& tp) {
        RawDataWriter(values, tp, TensorProto_DataType_UINT8);
      },
      RawDataChecker<uint8_t>);

  // Test constant node conversion for SparseTensor with external data
  PathString tensor_filename(ORT_TSTR("tensor_XXXXXX"));
  TestConversion<float>(
      true,
      [&tensor_filename](const std::vector<float>& values, TensorProto& tp) {
        CreateTensorWithExternalData<float>(TensorProto_DataType_FLOAT, values, tensor_filename, tp);
      },
      RawDataChecker<float>);
  DeleteFileFromDisk(tensor_filename.c_str());
}

/// Dense to Sparse conversion tests
#if !defined(ORT_MINIMAL_BUILD)

template <typename T>
static std::vector<T> CreateSparseValues() {
  return {0, 2, 3, 0};
}

/* std::string support in the future
template <>
std::vector<std::string> CreateSparseValues<std::string>() {
  return {"", "two", "three", ""};
}
*/

template <>
std::vector<BFloat16> CreateSparseValues<BFloat16>() {
  return {BFloat16(0.f), BFloat16(2.f), BFloat16(3.f), BFloat16(0.f)};
}

template <>
std::vector<MLFloat16> CreateSparseValues<MLFloat16>() {
  return {MLFloat16(0.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(0.f)};
}

template <typename T>
static std::vector<T> CreateSparseValuesAllZeros() {
  return {0, 0, 0, 0};
}

template <>
std::vector<BFloat16> CreateSparseValuesAllZeros<BFloat16>() {
  return {BFloat16(0.f), BFloat16(0.f), BFloat16(0.f), BFloat16(0.f)};
}

template <>
std::vector<MLFloat16> CreateSparseValuesAllZeros<MLFloat16>() {
  return {MLFloat16(0.f), MLFloat16(0.f), MLFloat16(0.f), MLFloat16(0.f)};
}

template <typename T>
TensorProto CreateDenseTensor(std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
                              std::vector<T>& expected_values, std::vector<int64_t>& expected_indicies) {
  TensorProto result;
  std::vector<T> values = CreateSparseValues<T>();
  expected_indicies = {1, 2};
  for (const auto& ind : expected_indicies) {
    expected_values.push_back(values[ind]);
  }
  inserter(values, result);
  result.add_dims(static_cast<int64_t>(values.size()));
  return result;
}

template <typename T>
TensorProto CreateDenseTensorAllZeros(std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter) {
  TensorProto result;
  std::vector<T> values = CreateSparseValuesAllZeros<T>();
  inserter(values, result);
  result.add_dims(static_cast<int64_t>(values.size()));
  return result;
}

int64_t ActualSize(const SparseTensorProto& actual) {
  int64_t actual_size = 1;
  for (const auto dim : actual.values().dims()) {
    actual_size *= dim;
  }
  return actual_size;
}

template <typename T>
static void RawSparseDataChecker(gsl::span<const T> expected_values,
                                 gsl::span<const int64_t> expected_indicies,
                                 const SparseTensorProto& actual) {
  const int64_t actual_size = ActualSize(actual);

  const T* raw_data = reinterpret_cast<const T*>(actual.values().raw_data().data());
  auto actual_span = gsl::make_span<const T>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected_values));

  // Check indicies
  EXPECT_THAT(actual.indices().data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto actual_indicies = gsl::make_span<const int64_t>(actual.indices().int64_data().data(), actual.indices().int64_data_size());
  EXPECT_THAT(actual_indicies, testing::ContainerEq(expected_indicies));
}

template <>
void RawSparseDataChecker<BFloat16>(gsl::span<const BFloat16> expected_bfloat,
                                    gsl::span<const int64_t> expected_indicies,
                                    const SparseTensorProto& actual) {
  const int64_t actual_size = ActualSize(actual);

  static_assert(sizeof(uint16_t) == sizeof(BFloat16), "Expecting equal sizes");
  auto expected = expected_bfloat.as_span<const uint16_t>();
  const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(actual.values().raw_data().data());
  auto actual_span = gsl::make_span<const uint16_t>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
  // Check indicies
  EXPECT_THAT(actual.indices().data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto actual_indicies = gsl::make_span<const int64_t>(actual.indices().int64_data().data(), actual.indices().int64_data_size());
  EXPECT_THAT(actual_indicies, testing::ContainerEq(expected_indicies));
}

template <>
void RawSparseDataChecker<MLFloat16>(gsl::span<const MLFloat16> expected_bfloat,
                                     gsl::span<const int64_t> expected_indicies,
                                     const SparseTensorProto& actual) {
  const int64_t actual_size = ActualSize(actual);

  static_assert(sizeof(uint16_t) == sizeof(MLFloat16), "Expecting equal sizes");
  auto expected = expected_bfloat.as_span<const uint16_t>();
  const uint16_t* raw_data = reinterpret_cast<const uint16_t*>(actual.values().raw_data().data());
  auto actual_span = gsl::make_span<const uint16_t>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
  // Check indicies
  EXPECT_THAT(actual.indices().data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto actual_indicies = gsl::make_span<const int64_t>(actual.indices().int64_data().data(), actual.indices().int64_data_size());
  EXPECT_THAT(actual_indicies, testing::ContainerEq(expected_indicies));
}

template <typename T>
static void TestDenseToSparseConversionValues(
    std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
    std::function<void(gsl::span<const T> expected,
                       gsl::span<const int64_t> expected_indicies,
                       const SparseTensorProto& actual)>
        checker) {
  std::vector<T> expected_values;
  std::vector<int64_t> expected_indicies;
  // Path is required for loading external data
  // Using empty path here since the data is not external
  Path model_path;
  TensorProto dense_tensor = CreateDenseTensor(inserter, expected_values, expected_indicies);

  SparseTensorProto sparse_tensor;
  utils::DenseTensorToSparseTensorProto(dense_tensor, model_path, sparse_tensor);

  gsl::span<const T>
      expected_values_span = gsl::make_span(expected_values.data(), expected_values.size());
  gsl::span<const int64_t> expected_ind_span = gsl::make_span(expected_indicies.data(), expected_indicies.size());
  checker(expected_values_span, expected_ind_span, sparse_tensor);
}

template <typename T>
static void TestDenseAllZerosToSparseConversion(
    std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
    std::function<void(gsl::span<const T> expected,
                       gsl::span<const int64_t> expected_indicies,
                       const SparseTensorProto& actual)>
        checker) {
  std::vector<T> expected_values;
  std::vector<int64_t> expected_indicies;
  // Path is required for loading external data
  // Using empty path here since the data is not external
  Path model_path;
  TensorProto dense_tensor = CreateDenseTensorAllZeros(inserter);

  SparseTensorProto sparse_tensor;
  utils::DenseTensorToSparseTensorProto(dense_tensor, model_path, sparse_tensor);

  gsl::span<const T>
      expected_values_span = gsl::make_span(expected_values.data(), expected_values.size());
  gsl::span<const int64_t> expected_ind_span = gsl::make_span(expected_indicies.data(), expected_indicies.size());
  checker(expected_values_span, expected_ind_span, sparse_tensor);
}

template <typename T>
static void TestDenseToSparseConversion(std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
                                        std::function<void(gsl::span<const T> expected,
                                                           gsl::span<const int64_t> expected_indicies,
                                                           const SparseTensorProto& actual)>
                                            checker) {
  TestDenseToSparseConversionValues<T>(inserter, checker);
  TestDenseAllZerosToSparseConversion<T>(inserter, checker);
}


TEST(SparseTensorConversionTests, TestDenseToSparseConversion) {
  TestDenseToSparseConversion<float>(
      [](const std::vector<float>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_FLOAT);
        tp.set_name("dense_float");
        tp.mutable_float_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<float>);

  TestDenseToSparseConversion<double>(
      [](const std::vector<double>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_DOUBLE);
        tp.set_name("dense_double");
        tp.mutable_double_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<double>);

  TestDenseToSparseConversion<BFloat16>(
      [](const std::vector<BFloat16>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_BFLOAT16);
        tp.set_name("dense_bfloat16");
        for (auto v : values) {
          tp.mutable_int32_data()->Add(v.val);
        }
      },
      RawSparseDataChecker<BFloat16>);

  TestDenseToSparseConversion<MLFloat16>(
      [](const std::vector<MLFloat16>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_FLOAT16);
        tp.set_name("dense_float16");
        for (auto v : values) {
          tp.mutable_int32_data()->Add(v.val);
        }
      },
      RawSparseDataChecker<MLFloat16>);

  TestDenseToSparseConversion<int16_t>(
      [](const std::vector<int16_t>& values, TensorProto& tp) {
        tp.set_name("dense_int16");
        tp.set_data_type(TensorProto_DataType_INT16);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<int16_t>);

  TestDenseToSparseConversion<uint16_t>(
      [](const std::vector<uint16_t>& values, TensorProto& tp) {
        tp.set_name("dense_uint16");
        tp.set_data_type(TensorProto_DataType_UINT16);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<uint16_t>);

  TestDenseToSparseConversion<int32_t>(
      [](const std::vector<int32_t>& values, TensorProto& tp) {
        tp.set_name("dense_int32");
        tp.set_data_type(TensorProto_DataType_INT32);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<int32_t>);

  TestDenseToSparseConversion<uint32_t>(
      [](const std::vector<uint32_t>& values, TensorProto& tp) {
        tp.set_name("dense_uint32");
        tp.set_data_type(TensorProto_DataType_UINT32);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<uint32_t>);

  TestDenseToSparseConversion<int64_t>(
      [](const std::vector<int64_t>& values, TensorProto& tp) {
        tp.set_name("dense_int64");
        tp.set_data_type(TensorProto_DataType_INT64);
        tp.mutable_int64_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<int64_t>);

  TestDenseToSparseConversion<uint64_t>(
      [](const std::vector<uint64_t>& values, TensorProto& tp) {
        tp.set_name("dense_uint64");
        tp.set_data_type(TensorProto_DataType_UINT64);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<uint64_t>);

  TestDenseToSparseConversion<int8_t>(
      [](const std::vector<int8_t>& values, TensorProto& tp) {
        tp.set_name("dense_int8");
        tp.set_data_type(TensorProto_DataType_INT8);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawSparseDataChecker<int8_t>);

  TestDenseToSparseConversion<uint8_t>(
      [](const std::vector<uint8_t>& values, TensorProto& tp) {
        tp.set_name("dense_int64");
        RawDataWriter(values, tp, TensorProto_DataType_UINT8);
      },
      RawSparseDataChecker<uint8_t>);
}

#endif  // !ORT_MINIMAL_BUILD

}  // namespace test
}  // namespace onnxruntime
