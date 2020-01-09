// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <memory>
#include "core/platform/env.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/session/onnxruntime_c_api.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"

using namespace onnxruntime;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {
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
  EXPECT_TRUE(graph.Resolve().IsOK());

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

TEST(ONNXModelsTest, squeeze_net) {
  // NOTE: this requires the current directory to be where onnxruntime_ir_UT.exe is located
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(ORT_TSTR("../models/opset8/test_squeezenet/model.onnx"), model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  TestResolve(model->MainGraph());
}
#endif

TEST(ONNXModelsTest, non_existing_model) {
  // NOTE: this requires the current directory to be where onnxruntime_ir_UT.exe is located
  std::shared_ptr<Model> model;
  common::Status st = Model::Load(ORT_TSTR("./testdata/non_existing_model_XXXXXX/model.onnx"), model, nullptr,
                                  DefaultLoggingManager().DefaultLogger());
  ASSERT_FALSE(st.IsOK());
  ASSERT_EQ(st.Code(), common::NO_SUCHFILE);
}

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
TEST(ONNXModelsTest1, bvlc_alexnet_1) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  ASSERT_TRUE(Env::Default().FileOpenRd(ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx"), fd).IsOK());
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
  ASSERT_TRUE(Env::Default().FileClose(fd).IsOK());

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx"), model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());

  // Check the graph input/output/value_info should have the same size as specified in the model file.
  EXPECT_EQ(model_proto.graph().value_info_size(), model->MainGraph().GetValueInfo().size());
  EXPECT_EQ(model_proto.graph().input_size(), model->MainGraph().GetInputs().size() + model->MainGraph().GetAllInitializedTensors().size());
  EXPECT_EQ(model_proto.graph().output_size(), model->MainGraph().GetOutputs().size());
  TestResolve(model->MainGraph());
}

class ONNXModelsTest : public ::testing::TestWithParam<const ORTCHAR_T*> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
 public:
  std::basic_string<ORTCHAR_T> GetModelFileName() const {
    std::basic_ostringstream<ORTCHAR_T> oss;
    oss << ORT_TSTR("../models/opset7/test_") << GetParam() << ORT_TSTR("/model.onnx");
    return oss.str();
  }
};

TEST_P(ONNXModelsTest, LoadFromFile) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(GetModelFileName(), model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  TestResolve(model->MainGraph());
}

TEST_P(ONNXModelsTest, LoadFromProtobuf) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  auto st = Env::Default().FileOpenRd(GetModelFileName(), fd);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  coded_input->SetTotalBytesLimit(INT_MAX);
  std::unique_ptr<ModelProto> model_proto = onnxruntime::make_unique<ModelProto>();
  bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  ASSERT_TRUE(result);
  ASSERT_TRUE(Env::Default().FileClose(fd).IsOK());
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(std::move(model_proto), model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  TestResolve(model->MainGraph());
}

#ifndef DISABLE_CONTRIB_OPS
INSTANTIATE_TEST_CASE_P(ONNXModelsTests,
                        ONNXModelsTest,
                        ::testing::Values(ORT_TSTR("bvlc_alexnet"), ORT_TSTR("bvlc_googlenet"), ORT_TSTR("bvlc_reference_caffenet"), ORT_TSTR("bvlc_reference_rcnn_ilsvrc13"), ORT_TSTR("densenet121"), ORT_TSTR("emotion_ferplus"), ORT_TSTR("inception_v1"), ORT_TSTR("inception_v2"), ORT_TSTR("mnist"), ORT_TSTR("resnet50"), ORT_TSTR("shufflenet"), ORT_TSTR("squeezenet"), ORT_TSTR("tiny_yolov2"), ORT_TSTR("vgg19"), ORT_TSTR("zfnet512")));
#else
INSTANTIATE_TEST_CASE_P(ONNXModelsTests,
                        ONNXModelsTest,
                        ::testing::Values(ORT_TSTR("bvlc_alexnet"), ORT_TSTR("bvlc_googlenet"), ORT_TSTR("bvlc_reference_caffenet"), ORT_TSTR("bvlc_reference_rcnn_ilsvrc13"), ORT_TSTR("densenet121"), ORT_TSTR("emotion_ferplus"), ORT_TSTR("inception_v1"), ORT_TSTR("inception_v2"), ORT_TSTR("mnist"), ORT_TSTR("resnet50"), ORT_TSTR("shufflenet"), ORT_TSTR("squeezenet"), ORT_TSTR("vgg19"), ORT_TSTR("zfnet512")));
#endif

#endif

// test a model that conforms to ONNX IR v4 where there are initializers that are not graph inputs.
// a NodeArg should be created for all initializers in this case.
// the test model contains initializers that are used as implicit inputs in a subgraph, and the NodeArg is required
// for Graph::Resolve to succeed when processing the subgraph.
TEST(ONNXModelsTest, TestIRv4NonInputInitializers) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(ORT_TSTR("testdata/subgraph_implicit_input_from_initializer.onnx"), model, nullptr,
                          DefaultLoggingManager().DefaultLogger())
                  .IsOK());
  EXPECT_TRUE(model->MainGraph().Resolve().IsOK());
}

// test a model that has an op with a FunctionBody and one of the nodes within the FunctionBody has a subgraph in it.
// The test model has is an opset-11 op with a 'Range' node.
// 'Range' has a FunctionBody and has a 'Loop' node with a subgraph.
// Graph::Resolve to succeed when processing the subgraph pertaining to the overall FunctionBody.
TEST(ONNXModelsTest, TestModelsWithAnOpContainingAFunctionBody) {
  std::shared_ptr<Model> model;

  auto status = Model::Load(ORT_TSTR("testdata/model_containing_op_with_function_body.onnx"), model, nullptr,
                            DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status;

  status = model->MainGraph().Resolve();
  EXPECT_TRUE(status.IsOK()) << status;
}

}  // namespace test
}  // namespace onnxruntime
