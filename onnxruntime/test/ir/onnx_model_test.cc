// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <memory>
#include "core/platform/env.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/session/onnxruntime_c_api.h"
#include "test/providers/provider_test_utils.h"  //For ASSERT_STATUS_OK
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"

using namespace onnxruntime;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {
class ONNXModelsTest : public ::testing::Test {
 protected:
  ONNXModelsTest() {
    logger_ = DefaultLoggingManager().CreateLogger("GraphTest");
  }

  std::unique_ptr<logging::Logger> logger_;
};
#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
// Tests that Resolve() properly clears the state of topological sorted nodes,
// inputs, outputs and valueInfo.
// Assumes the graph passed in has been previously resolved.
static void TestResolve(onnxruntime::Graph& graph) {
  GraphViewer graph_viewer(graph);
  auto& nodes_before = graph_viewer.GetNodesInTopologicalOrder();
  auto& inputs_before = graph.GetInputs();
  auto& outputs_before = graph.GetOutputs();
  auto& value_info_before = graph.GetValueInfo();

  // Touch the graph to force Resolve() to recompute.
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  ASSERT_STATUS_OK(graph.Resolve());

  GraphViewer graph_viewer_2(graph);
  auto& nodes_after = graph_viewer_2.GetNodesInTopologicalOrder();
  auto& inputs_after = graph.GetInputs();
  auto& outputs_after = graph.GetOutputs();
  auto& value_info_after = graph.GetValueInfo();

  // Multiple calls to Resolve() should not alter the sorted nodes,
  // inputs, outputs and valueInfo. The internal state should be
  // cleared.
  EXPECT_EQ(nodes_before, nodes_after);
  EXPECT_EQ(inputs_before, inputs_after);
  EXPECT_EQ(outputs_before, outputs_after);
  EXPECT_EQ(value_info_before, value_info_after);
}

TEST_F(ONNXModelsTest, squeeze_net) {
  // NOTE: this requires the current directory to be where onnxruntime_ir_UT.exe is located
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("../models/opset8/test_squeezenet/model.onnx"), model, nullptr, *logger_));
  TestResolve(model->MainGraph());
}
#endif

TEST_F(ONNXModelsTest, non_existing_model) {
  // NOTE: this requires the current directory to be where onnxruntime_ir_UT.exe is located
  std::shared_ptr<Model> model;
  common::Status st = Model::Load(ORT_TSTR("./testdata/non_existing_model_XXXXXX/model.onnx"), model, nullptr, *logger_);
  ASSERT_FALSE(st.IsOK());
  ASSERT_EQ(st.Code(), common::NO_SUCHFILE);
}

TEST_F(ONNXModelsTest, future_opset) {
  // NOTE: this requires the current directory to be where onnxruntime_ir_UT.exe is located
  std::shared_ptr<Model> model;
  common::Status st = Model::Load(ORT_TSTR("./testdata/add_opset_314159.onnx"), model, nullptr, *logger_);
  ASSERT_FALSE(st.IsOK());
}

class ONNXModelsTest1 : public ::testing::TestWithParam<const ORTCHAR_T*> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
 public:
  ONNXModelsTest1() {
    logger_ = DefaultLoggingManager().CreateLogger("GraphTest");
  }

  std::unique_ptr<logging::Logger> logger_;
  std::basic_string<ORTCHAR_T> GetModelFileName() const {
    std::basic_ostringstream<ORTCHAR_T> oss;
    oss << ORT_TSTR("../models/opset7/test_") << GetParam() << ORT_TSTR("/model.onnx");
    return oss.str();
  }
};
#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
TEST_F(ONNXModelsTest, bvlc_alexnet_1) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  ASSERT_STATUS_OK(Env::Default().FileOpenRd(ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx"), fd));
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX);
  ModelProto model_proto;
  bool result = model_proto.ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  EXPECT_TRUE(result);
  ASSERT_STATUS_OK(Env::Default().FileClose(fd));

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx"), model, nullptr,
                               *logger_));

  // Check the graph input/output/value_info should have the same size as specified in the model file.
  EXPECT_EQ(static_cast<size_t>(model_proto.graph().value_info_size()), model->MainGraph().GetValueInfo().size());
  EXPECT_EQ(static_cast<size_t>(model_proto.graph().input_size()), model->MainGraph().GetInputs().size() + model->MainGraph().GetAllInitializedTensors().size());
  EXPECT_EQ(static_cast<size_t>(model_proto.graph().output_size()), model->MainGraph().GetOutputs().size());
  TestResolve(model->MainGraph());
}

TEST_P(ONNXModelsTest1, LoadFromFile) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(GetModelFileName(), model, nullptr,
                               *logger_));
  TestResolve(model->MainGraph());
}

TEST_P(ONNXModelsTest1, LoadFromProtobuf) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  ASSERT_STATUS_OK(Env::Default().FileOpenRd(GetModelFileName(), fd));
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  coded_input->SetTotalBytesLimit(INT_MAX);
  ModelProto model_proto;
  bool result = model_proto.ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  ASSERT_TRUE(result);
  ASSERT_STATUS_OK(Env::Default().FileClose(fd));
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr,
                               *logger_));
  TestResolve(model->MainGraph());
}

#ifndef DISABLE_CONTRIB_OPS
INSTANTIATE_TEST_SUITE_P(ONNXModelsTests,
                        ONNXModelsTest1,
                        ::testing::Values(ORT_TSTR("bvlc_alexnet"), ORT_TSTR("bvlc_googlenet"), ORT_TSTR("bvlc_reference_caffenet"), ORT_TSTR("bvlc_reference_rcnn_ilsvrc13"), ORT_TSTR("densenet121"), ORT_TSTR("emotion_ferplus"), ORT_TSTR("inception_v1"), ORT_TSTR("inception_v2"), ORT_TSTR("mnist"), ORT_TSTR("resnet50"), ORT_TSTR("shufflenet"), ORT_TSTR("squeezenet"), ORT_TSTR("tiny_yolov2"), ORT_TSTR("vgg19"), ORT_TSTR("zfnet512")));
#else
INSTANTIATE_TEST_SUITE_P(ONNXModelsTests,
                        ONNXModelsTest1,
                        ::testing::Values(ORT_TSTR("bvlc_alexnet"), ORT_TSTR("bvlc_googlenet"), ORT_TSTR("bvlc_reference_caffenet"), ORT_TSTR("bvlc_reference_rcnn_ilsvrc13"), ORT_TSTR("densenet121"), ORT_TSTR("emotion_ferplus"), ORT_TSTR("inception_v1"), ORT_TSTR("inception_v2"), ORT_TSTR("mnist"), ORT_TSTR("resnet50"), ORT_TSTR("shufflenet"), ORT_TSTR("squeezenet"), ORT_TSTR("vgg19"), ORT_TSTR("zfnet512")));
#endif

#endif

// test a model that conforms to ONNX IR v4 where there are initializers that are not graph inputs.
// a NodeArg should be created for all initializers in this case.
// the test model contains initializers that are used as implicit inputs in a subgraph, and the NodeArg is required
// for Graph::Resolve to succeed when processing the subgraph.
TEST_F(ONNXModelsTest, TestIRv4NonInputInitializers) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/subgraph_implicit_input_from_initializer.onnx"), model, nullptr,
                               *logger_));
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
}

// test a model that has an op with a FunctionBody and one of the nodes within the FunctionBody has a subgraph in it.
// The test model has is an opset-11 op with a 'Range' node.
// 'Range' has a FunctionBody and has a 'Loop' node with a subgraph.
// Graph::Resolve to succeed when processing the subgraph pertaining to the overall FunctionBody.
TEST_F(ONNXModelsTest, TestModelsWithAnOpContainingAFunctionBody) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/model_containing_op_with_function_body.onnx"), model, nullptr,
                               *logger_));
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
}

// The following tests verify ORT can successfully load models which reference functions
// present in the ModelProto aka model local functions. This feature was added to ONNX standard starting IRv8

void BuildFunction(FunctionProto& function_proto, 
    const std::string& name, const std::string& domain, 
    const std::vector<NodeProto>& nodes, 
    const std::vector<std::string>& inputs, 
    const std::vector<std::string>& outputs, 
    const std::unordered_map<std::string, int>& opset_imports) {
  for (const auto& node : nodes) {
    auto new_node = function_proto.add_node();
    new_node->CopyFrom(node);
  }

  function_proto.set_name(name);
  function_proto.set_domain(domain);
  function_proto.set_doc_string("Test function proto");

  for (auto& input : inputs)
    function_proto.add_input(input);

  for (auto& output : outputs)
    function_proto.add_output(output);

  for (auto& opset_import : opset_imports) {
    auto* func_opset_import = function_proto.mutable_opset_import()->Add();
    func_opset_import->set_domain(opset_import.first);
    func_opset_import->set_version(opset_import.second);
  }
}

void BuildFunctionFoo(FunctionProto& function_proto, const std::string& domain) {
  auto func_body_nodes = FunctionBodyHelper::BuildNodes(
      {// nodes: {outputs, op, inputs, attributes}
       FunctionBodyHelper::Const<float>("Q_Min", 0.f),
       FunctionBodyHelper::Const<float>("Q_Max", 255.f),
       {{"X_Min"}, "ReduceMin", {"x"}, {MakeAttribute("keepdims", int64_t(0))}},
       {{"X_Max"}, "ReduceMax", {"x"}, {MakeAttribute("keepdims", int64_t(0))}},
       {{"X_Range"}, "Sub", {"X_Max", "X_Min"}},
       {{"s"}, "Div", {"X_Range", "Q_Max"}},
       {{"zp_fp"}, "Sub", {"Q_Min", "s"}},
       {{"zp"}, "Cast", {"zp_fp"}, {MakeAttribute("to", int64_t(2))}},
       {{"y"}, "QuantizeLinear", {"x", "s", "zp"}}});

  BuildFunction(function_proto, "foo", domain, func_body_nodes, {"x"}, {"y"}, {{"", 13}});
}

void RunFunctionTests(ModelProto&& model_proto) {
  std::shared_ptr<Model> model;
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {registry};
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, &regs,
                               *(DefaultLoggingManager().CreateLogger("GraphTest"))));

  // Test function inline
  auto& graph = model->MainGraph();
  bool function_inlined = false;
  do {
    function_inlined = false;
    for (auto& node : graph.Nodes()) {
      if (node.GetFunctionBody() != nullptr) {
        ASSERT_STATUS_OK(graph.InlineFunction(node));
        function_inlined = true;
        break;
      }
    }
  } while (function_inlined);

  ASSERT_STATUS_OK(graph.Resolve());
}

// Tests:
// 1. Function initialization and inlining.
// 2. Input\output name handling when intermediate function body node input\outputs have same names as outer graph.
// 3. Input\output name handling when function body input output names don't match node input output names
TEST(FunctionVerification, TestModelLocalFunctions) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x) => (uint8[N] s)
{
    t = custom_domain.foo(x)
    s = Identity(t)
}
)ONNX";

  ModelProto model_proto;
  ONNX_NAMESPACE::OnnxParser parser(code);
  auto status = parser.Parse(model_proto);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(parser.EndOfInput());

  auto* function_proto = model_proto.mutable_functions()->Add();
  BuildFunctionFoo(*function_proto, "custom_domain");

  RunFunctionTests(std::move(model_proto));
}

// Tests Input\Output name handling where function output is consumed by function body node as well.
// This is treated as a special case because we need to test that the node arg name is
// handled properly. Specially when this function output is also remapped to node output.
TEST(FunctionVerification, TestModelLocalFunctionsWithMultipleOutputs) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x) => (float[N] out)
{
    o1, o2 = custom_domain.bar(x)
    out = Add(o1, o2)
}
)ONNX";

  ModelProto model_proto;
  ONNX_NAMESPACE::OnnxParser parser(code);
  auto status = parser.Parse(model_proto);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(parser.EndOfInput());

  auto function_proto = model_proto.mutable_functions()->Add();
  auto func_body_nodes = FunctionBodyHelper::BuildNodes(
      {// nodes: {outputs, op, inputs, attributes, domain}
       {{"o2"}, "Identity", {"x"}},
       {{"o1"}, "Identity", {"o2"}}});
  BuildFunction(*function_proto, "bar", "custom_domain",
                func_body_nodes, {"x"}, {"o1", "o2"}, {{"", 13}});

  RunFunctionTests(std::move(model_proto));
}

// Tests:
// 1. Nested functions initialization and inlining.
// 1. Input\output name handling when intermediate function body node input\outputs have same names as outer graph.
// 2. Input\output name handling when function body input output names don't match node input output names
TEST(FunctionVerification, TestNestedModelLocalFunctions) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x) => (uint8[N] zp)
{
    c = custom_domain.foo(x)
    zp = Identity(c)
}
)ONNX";

  ModelProto model_proto;
  ONNX_NAMESPACE::OnnxParser parser(code);
  auto status = parser.Parse(model_proto);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(parser.EndOfInput());

  auto* function_proto = model_proto.mutable_functions()->Add();
  BuildFunctionFoo(*function_proto, "custom_domainA");

  // Build second function proto
  // intentionally using same function name to test 
  // that domainA:name and domainB:name are allowed.
  function_proto = model_proto.mutable_functions()->Add();
  auto func_body_nodes = FunctionBodyHelper::BuildNodes(
      {// nodes: {outputs, op, inputs, attributes, domain}
       {{"out"}, "foo", {"x"}, {}, "custom_domainA"},
       {{"s"}, "Identity", {"out"}}});
  BuildFunction(*function_proto, "foo", "custom_domain",
                func_body_nodes, {"x"}, {"s"}, {{"", 13}, {"custom_domainA", 1}});

  RunFunctionTests(std::move(model_proto));
}

// Tests:
// 1. Function initialization and inlining when there are multiple references to the same function
// from within a function body and directly from a graph
// 2. Input\output and node names are handled correctly (.i.e unique names are generated where necessary) when inlining the 
// same function multiple times in the graph.
// 3. Unique names are generated for intermediate node input\outputs when they match the names of node input\outputs
TEST(FunctionVerification, TestNestedModelLocalFunctionsWithMultipleReferences) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain" : 1, "custom_domainA" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x, float[N] y) => (float[N] zp)
{
    c = custom_domain.bar(x)
    zp1 = Cast<to = 1>(c)
    d = custom_domainA.foo(y)
    zp2 = Cast<to = 1>(d)
    zp = Sub(zp1, zp2)
}
)ONNX";

  ModelProto model_proto;
  ONNX_NAMESPACE::OnnxParser parser(code);
  auto status = parser.Parse(model_proto);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(parser.EndOfInput());

  auto* function_proto = model_proto.mutable_functions()->Add();
  BuildFunctionFoo(*function_proto, "custom_domainA");

  // Build second function proto
  // intentionally using same function name to test
  // that domainA:name and domainB:name are allowed.
  function_proto = model_proto.mutable_functions()->Add();
  auto func_body_nodes = FunctionBodyHelper::BuildNodes(
      {// nodes: {outputs, op, inputs, attributes, domain}
       {{"s"}, "foo", {"x"}, {}, "custom_domainA"},
       {{"out"}, "Identity", {"s"}}});
  BuildFunction(*function_proto, "bar", "custom_domain", func_body_nodes,
                {"x"}, {"out"}, {{"", 13}, {"custom_domainA", 1}});

  RunFunctionTests(std::move(model_proto));
}

}  // namespace test
}  // namespace onnxruntime
