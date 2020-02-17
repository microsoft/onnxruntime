// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST(GraphTransformationTests, GistEncodeDecode) {
  auto model_uri = MODEL_FOLDER "../test_training_model.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleGistTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<GistEncodeDecode>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["GistBinarizeEncoder"] == op_to_count["GistBinarizeEncoder"]);
}

}  // namespace test
}  // namespace onnxruntime
