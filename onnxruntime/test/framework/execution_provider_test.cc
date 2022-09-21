// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/graph/model.h"
#include "test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

#include <fstream>

namespace onnxruntime {
namespace test {

class TestEP : public IExecutionProvider {
  static constexpr const char* kEPType = "TestEP";

 public:
  TestEP() : IExecutionProvider{kEPType, true} {}

  int GetId(const GraphViewer& viewer, HashValue& model_hash) {
    return GenerateMetaDefId(viewer, model_hash);
  }
};

TEST(ExecutionProviderTest, MetadefIdGeneratorUsingModelPath) {
  TestEP ep;

  auto test_model = [&ep](const std::basic_string<ORTCHAR_T>& model_path) {
    std::shared_ptr<Model> model;
    ASSERT_TRUE(Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

    Graph& graph = model->MainGraph();
    GraphViewer viewer(graph);

    // check for stable non-zero model_hash, and incrementing id.
    HashValue model_hash;
    int id = ep.GetId(viewer, model_hash);
    ASSERT_EQ(id, 0);
    ASSERT_NE(model_hash, 0);

    for (int i = 1; i < 4; ++i) {
      HashValue cur_model_hash;
      int cur_id = ep.GetId(viewer, cur_model_hash);
      ASSERT_EQ(cur_id, i);
      ASSERT_EQ(cur_model_hash, model_hash);
    }
  };

  test_model(ORT_TSTR("testdata/mnist.onnx"));
  // load a new model instance and check it has a separate scope for the generated ids
  test_model(ORT_TSTR("testdata/ort_github_issue_4031.onnx"));
}

// test when the model hash is created by hashing the contents of the main graph instead of the model path
TEST(ExecutionProviderTest, MetadefIdGeneratorUsingModelHashing) {
  TestEP ep;

  auto model_path = ORT_TSTR("testdata/mnist.onnx");

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  Graph& graph = model->MainGraph();
  GraphViewer viewer(graph);

  // get the hash for the model when loaded from file
  HashValue model_hash;
  int id = ep.GetId(viewer, model_hash);
  ASSERT_EQ(id, 0);
  ASSERT_NE(model_hash, 0);

  // now load the model from bytes and check the hash differs
  std::ifstream model_file_stream(model_path, std::ios::in | std::ios::binary);

  std::shared_ptr<Model> model2;
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(model_file_stream, &model_proto));
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), PathString(), model2, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  Graph& graph2 = model2->MainGraph();
  GraphViewer viewer2(graph2);

  HashValue model_hash2;
  int id2 = ep.GetId(viewer2, model_hash2);
  ASSERT_EQ(id2, 0) << "Id for new model should always start at zero";
  ASSERT_NE(model_hash, model_hash2) << "Hash from model path should differ from hash based on model contents";
}

}  // namespace test
}  // namespace onnxruntime
