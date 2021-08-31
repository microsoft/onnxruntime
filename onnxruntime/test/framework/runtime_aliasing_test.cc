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
  // This file tests the runtime aliasing feature.
  // It includes a mock op, MayIncrement that uses this feature.


static SessionOptions GetSessionOptions() {
  SessionOptions session_options;
  session_options.session_log_severity_level = 0;
  session_options.session_log_verbosity_level = 100;
  return session_options;

}
class RuntimeAliasingTest : public testing::Test {
 protected:
  InferenceSession session_object;
  std::shared_ptr<CustomRegistry> registry;
  std::unique_ptr<Model> model;
  std::vector<TypeProto> types;
  std::vector<OpSchema> schemas;
  NameMLValMap feeds;

  std::vector<std::string> output_names;
  std::vector<OrtValue> expected_output;

  // The ONNX schema for MayIncrement:
  static ONNX_NAMESPACE::OpSchema MayIncrementSchema() {
    ONNX_NAMESPACE::OpSchema schema;

    schema.SetName("MayIncrement")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(10)
      .SetDoc(R"DOC(
If the first element of increment is 1, each value in input will be incremented. Otherwise, this is a noop.
)DOC")
    .Input(
        0,
        "input",
        "Input tensor",
        "T",
        OpSchema::Single)
    .Input(
        1,
        "increment",
        "Whether to increment or not",
        "B",
        OpSchema::Single)
    .Output(
        0,
        "output",
        "Output tensor",
        "T",
        OpSchema::Single)
    .TypeConstraint(
        "T",
        {"tensor(int64)"},
        "Input and Output type")
    .TypeConstraint(
        "B",
        {"tensor(int64)"},
        "Boolean flag type");

    return schema;
  }

  // A kernel implementation of MayIncrement:
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 2, "Expecting 2 inputs");

      const Tensor* input = ctx->Input<Tensor>(0);
      const Tensor* increment_tensor = ctx->Input<Tensor>(1);

      bool increment = increment_tensor->Data<int64_t>()[0] == 1;

      std::cout << "Executing kernel on data " << input->DataRaw() << std::endl;


      Tensor* output;

      if (!increment) {
        // try to alias the output
        output = ctx->AliasedOutput(0, 0);
        if (output->DataRaw() == input->DataRaw()) {
          return Status::OK();
        }
      }

      auto& shape = input->Shape();
      int64_t num_elements = input->Shape().Size();

      auto* input_values = input->Data<int64_t>();
      output = ctx->Output(0, shape);
      std::cout << "Created output with data " << output->DataRaw() << std::endl;

      auto* output_values = output->MutableData<int64_t>();

      for (int i = 0; i < num_elements; i++) {
        output_values[i] = input_values[i] + 1;
      }
      return Status::OK();
    }
  };

  // A KernelDefBuilder for MayIncrement:
  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName("MayIncrement")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(10)
      .Provider(onnxruntime::kCpuExecutionProvider)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .MayAlias(0, 0);
    return def;
  }


  RuntimeAliasingTest() : 
    session_object(GetSessionOptions(), GetEnvironment()),
                        registry(std::make_shared<CustomRegistry>()) {
    EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

    schemas.push_back(MayIncrementSchema());
    EXPECT_TRUE(registry->RegisterOpSet({schemas}, onnxruntime::kMLDomain, 10, 11).IsOK());

    // register the kernel
    auto kernel_def_builder = KernelDef();
    KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) {
      return new OpKernelImpl(info); };
    EXPECT_TRUE(registry->RegisterCustomKernel(kernel_def_builder, kernel_create_fn).IsOK());

    // build the model
    IOnnxRuntimeOpSchemaRegistryList custom_schema_registries = {registry->GetOpschemaRegistry()};
    model.reset(new Model("RuntimeAliasingTest", false, ModelMetaData(), PathString(), custom_schema_registries,
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

  OrtValue Constant(const std::vector<int64_t>& elts) {
    const std::vector<int64_t> shape{static_cast<int64_t>(elts.size())};
    return Constant(elts, shape);
  }

  OrtValue Constant(const std::vector<int64_t>& elts, const std::vector<int64_t>& shape) {
    OrtValue mlvalue;
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape, elts, &mlvalue);
    return mlvalue;
  }


  void AddInput(NodeArg* arg, const std::vector<int64_t>& value) {
    feeds[arg->Name()] = Constant(value);
  }

  void AddInput(NodeArg* arg, const std::vector<int64_t>& value, const std::vector<int64_t>& shape) {
    feeds[arg->Name()] = Constant(value, shape);
  }

  void ExpectEq(OrtValue val1, OrtValue val2) {
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

  void ExpectEq(OrtValue val1, const std::vector<int64_t>& data2) {
    // Restricted to case where val1 is an int64_t tensor
    auto& tensor1 = val1.Get<Tensor>();
    EXPECT_EQ(static_cast<uint64_t>(tensor1.Shape().Size()), data2.size());
    auto* data1 = tensor1.Data<int64_t>();
    for (int64_t i = 0, limit = tensor1.Shape().Size(); i < limit; ++i) {
      EXPECT_EQ(data1[i], data2[i]);
    }
  }

  void ExpectOutput(NodeArg* arg, const std::vector<int64_t>& value) {
    output_names.push_back(arg->Name());
    expected_output.push_back(Constant(value));
  }

  void RunTest() {
    RunOptions run_options;
    std::vector<OrtValue> fetches;

    EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
    std::cout << "done executing session" << std::endl;

    ASSERT_EQ(expected_output.size(), fetches.size());
    for (size_t i = 0; i < fetches.size(); ++i) {
      ExpectEq(fetches[i], expected_output[i]);
    }
  }
};

TEST_F(RuntimeAliasingTest, TestNoAliasingTrue) {
  auto data1 = Dense("data1");
  auto increment1 = Dense("increment1");
  auto maybe_incremented1 = Dense("maybe_incremented1");

  Node("MayIncrement", {data1, increment1}, {maybe_incremented1});

  // Check graph, serialize it and deserialize it back
  Graph& graph = model->MainGraph();
  Status status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  SerializeAndLoad();

  // Run the model
  AddInput(data1, {0, 1, 2, 3});
  AddInput(increment1, {1});
  ExpectOutput(maybe_incremented1, {1, 2, 3, 4});
  RunTest();
}


TEST_F(RuntimeAliasingTest, TestNoAliasingFalse) {
  auto data1 = Dense("data1");
  auto inc_true = Dense("increment_true");
  auto inc_false = Dense("increment_false");

  auto maybe_incremented1 = Dense("maybe_incremented1");
  auto maybe_incremented2 = Dense("maybe_incremented2");
  auto maybe_incremented3 = Dense("maybe_incremented3");

  Node("MayIncrement", {data1, inc_true}, {maybe_incremented1});
  Node("MayIncrement", {maybe_incremented1, inc_false}, {maybe_incremented2});
  Node("MayIncrement", {maybe_incremented2, inc_true}, {maybe_incremented3});

  // Check graph, serialize it and deserialize it back
  Graph& graph = model->MainGraph();
  Status status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  SerializeAndLoad();

  // Run the model
  AddInput(data1, {0, 1, 2, 3});
  AddInput(inc_true, {1});
  AddInput(inc_false, {0});
  ExpectOutput(maybe_incremented3, {2, 3, 4, 5});


  const SequentialExecutionPlan* p_seq_exec_plan = 
    session_object.GetSessionState().GetExecutionPlan();
  std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo{p_seq_exec_plan, &session_object.GetSessionState()};
  std::cout << "PLAN" << std::endl;
  std::cout << planinfo << std::endl;

  RunTest();
}

}  // namespace test
}  // namespace onnxruntime
