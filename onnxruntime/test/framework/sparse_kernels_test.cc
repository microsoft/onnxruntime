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
  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
  std::unordered_map<std::string, int> domain_to_version;
  Model model;
  Graph& graph;

  std::vector<OpSchema> schemas;
  std::vector<Action> register_actions;
  std::vector<TypeProto> types;

 public:
  SparseTensorTests() : session_object(SessionOptions(), GetEnvironment()),
                        registry(std::make_shared<CustomRegistry>()),
                        custom_schema_registries_{registry->GetOpschemaRegistry()},
                        domain_to_version{{onnxruntime::kMLDomain, 10}},
                        model("SparseTensorTest", false, ModelMetaData(), PathString(), custom_schema_registries_,
                              domain_to_version, {}, DefaultLoggingManager().DefaultLogger()),
                        graph(model.MainGraph()) {
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

  void SerializeAndLoad() {
    // Serialize model and deserialize it back
    std::string serialized_model;
    auto model_proto = model.ToProto();
    EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
    std::stringstream sstr(serialized_model);
    EXPECT_TRUE(session_object.Load(sstr).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());
  }

  NodeArg* Sparse(std::string name) {
    types.push_back(*DataTypeImpl::GetSparseTensorType<int64_t>()->GetTypeProto());
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  NodeArg* Dense(std::string name) {
    types.push_back(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  void Node(std::string op, const std::vector<NodeArg*> inputs, const std::vector<NodeArg*> outputs) {
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
  EXPECT_TRUE(graph.Resolve().IsOK());
  SerializeAndLoad();

  // Run the model
  AddInput(values, {-99, 2});
  AddInput(indices, {1, 1, 4, 4}, {2, 2});
  AddInput(shape, {5, 5});
  ExpectOutput(output, {99, 2});
  RunTest();
}

template <typename T>
static std::vector<T> CreateValues() {
  return {1, 2, 3, 4};
}

template <>
std::vector<std::string> CreateValues<std::string>() {
  return {"one", "two", "three", "four"};
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
static void TestConversion(bool use_1D_indices,
                           std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
                           std::function<void(gsl::span<const T> expected, const TensorProto& actual)> checker) {
  std::vector<T> expected;
  auto node = CreateConstantNode<T>(use_1D_indices, inserter, expected);

  TensorProto dense;
  utils::ConstantNodeProtoToTensorProto(node, dense);

  gsl::span<const T> expected_span = gsl::make_span<const T>(expected.data(), expected.size());
  checker(expected_span, dense);
}

template <typename T>
static void TestConversion(
    std::function<void(const std::vector<T>& values, TensorProto& tp)> inserter,
    std::function<void(gsl::span<const T> expected, const TensorProto& actual)> checker) {
  TestConversion(true, inserter, checker);
  TestConversion(false, inserter, checker);
}

template <typename T>
static void RawDataWriter(const std::vector<T>& values, TensorProto& tp, TensorProto_DataType datatype) {
  tp.set_data_type(datatype);
  tp.set_raw_data(values.data(), values.size() * sizeof(T));
}

template <typename T>
static void RawDataChecker(gsl::span<const T> expected, const TensorProto& actual) {
  int64_t actual_size = 1;
  for (const auto dim : actual.dims()) {
    actual_size *= dim;
  }

  const T* raw_data = reinterpret_cast<const T*>(actual.raw_data().data());
  auto actual_span = gsl::make_span<const T>(raw_data, actual_size);

  EXPECT_THAT(actual_span, testing::ContainerEq(expected));
}

TEST(SparseTensorConversionTests, TestConstantNodeConversion) {
  TestConversion<float>(
      [](const std::vector<float>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_FLOAT);
        tp.mutable_float_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<float>);

  TestConversion<int32_t>(
      [](const std::vector<int32_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT32);
        tp.mutable_int32_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int32_t>);

  TestConversion<int64_t>(
      [](const std::vector<int64_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_INT64);
        tp.mutable_int64_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<int64_t>);

  TestConversion<double>(
      [](const std::vector<double>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_DOUBLE);
        tp.mutable_double_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<double>);

  TestConversion<uint32_t>(
      [](const std::vector<uint32_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_UINT32);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());  // stored in uint64_data despite being uint32_t
      },
      RawDataChecker<uint32_t>);

  TestConversion<uint64_t>(
      [](const std::vector<uint64_t>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_UINT64);
        tp.mutable_uint64_data()->Add(values.cbegin(), values.cend());
      },
      RawDataChecker<uint64_t>);

  // test a couple of types with values in raw data field
  TestConversion<float>(
      [](const std::vector<float>& values, TensorProto& tp) {
        RawDataWriter(values, tp, TensorProto_DataType_FLOAT);
      },
      RawDataChecker<float>);

  TestConversion<int64_t>(
      [](const std::vector<int64_t>& values, TensorProto& tp) {
        RawDataWriter(values, tp, TensorProto_DataType_INT64);
      },
      RawDataChecker<int64_t>);

  // strings can't use raw data, and string_data is a RepeatedPtrField (vs. RepeatedField for simple types)
  // so has to be handled differently
  TestConversion<std::string>(
      [](const std::vector<std::string>& values, TensorProto& tp) {
        tp.set_data_type(TensorProto_DataType_STRING);
        for (auto cur = values.cbegin(), end = values.cend(); cur < end; ++cur) {
          tp.mutable_string_data()->Add(std::string(*cur));
        }
      },
      [](gsl::span<const std::string> expected, const TensorProto& actual) {
        const auto& actual_strings = actual.string_data();
        for (int64_t i = 0, end = expected.size(); i < end; ++i) {
          EXPECT_EQ(actual_strings[static_cast<int32_t>(i)], expected[i]);
        }
      });
}
}  // namespace test
}  // namespace onnxruntime
