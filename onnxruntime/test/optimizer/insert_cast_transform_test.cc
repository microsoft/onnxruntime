// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/insert_cast_transformer.h"
#include "core/graph/model.h"
#include "asserts.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(InsertCastTransformerTests, InsertCastNodeTwice) {
  auto model_uri = ORT_TSTR("testdata/transform/insert_Cast_twice.onnx");
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  InsertCastTransformer *insert_cast_transformer = new InsertCastTransformer("CastFloat16Transformer");
  
  // First insert
  bool modified = false;
  ASSERT_STATUS_OK(insert_cast_transformer->Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(modified);
  ASSERT_TRUE(op_to_count["Cast"] == 5);
  
  // Second insert
  modified = false;
  ASSERT_STATUS_OK(insert_cast_transformer->Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  op_to_count = CountOpsInGraph(graph);
  // Same graph without modification
  ASSERT_TRUE(!modified);
  ASSERT_TRUE(op_to_count["Cast"] == 5);

}
}  // namespace test
}  // namespace onnxruntime