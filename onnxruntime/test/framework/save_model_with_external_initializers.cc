// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <map>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/data_types.h"
#include "core/graph/model.h"
#include "core/graph/model_saving_options.h"
#include "core/framework/tensorprotoutils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime;

namespace onnxruntime {
namespace test {

Status LoadSaveAndCompareModel(const std::filesystem::path& input_onnx,
                               const std::filesystem::path& input_external_init_file,
                               const std::filesystem::path& output_onnx,
                               const std::filesystem::path& output_external_init_file,
                               const ModelSavingOptions& model_saving_options) {
  auto logger = DefaultLoggingManager().CreateLogger("LoadSaveAndCompareModel");
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(input_onnx, model, nullptr, *logger));
  std::filesystem::remove(output_onnx);
  std::filesystem::remove(output_external_init_file);
  ORT_RETURN_IF_ERROR(Model::SaveWithExternalInitializers(*model, output_onnx, output_external_init_file,
                                                          model_saving_options));

  std::shared_ptr<Model> model_from_external;
  ORT_RETURN_IF_ERROR(Model::Load(output_onnx.native(), model_from_external, nullptr, *logger));

  Graph& graph = model->MainGraph();
  // Perform shape inference on the graph, if this succeeds then it means that we could correctly read the
  // integer initializers used by reshape and transpose.
  ORT_RETURN_IF_ERROR(graph.Resolve());
  Graph& graph_from_external = model_from_external->MainGraph();

  InitializedTensorSet initializers = graph.GetAllInitializedTensors();
  InitializedTensorSet initializers_from_external = graph_from_external.GetAllInitializedTensors();

  ORT_RETURN_IF_NOT(initializers.size() == initializers_from_external.size(), "size mismatch");

  // Compare the initializers of the two versions.
  std::filesystem::path model_path{};
  std::filesystem::path external_data_path{};
  for (const auto& i : initializers) {
    const std::string kInitName = i.first;
    const ONNX_NAMESPACE::TensorProto* tensor_proto = i.second;
    const ONNX_NAMESPACE::TensorProto* from_external_tensor_proto = initializers_from_external[kInitName];

    std::vector<uint8_t> tensor_proto_data;
    model_path = input_onnx;
    external_data_path = (!input_external_init_file.empty()) ? (model_path.parent_path() / input_external_init_file) : std::filesystem::path();
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*tensor_proto, external_data_path, tensor_proto_data));
    size_t tensor_proto_size = tensor_proto_data.size();

    std::vector<uint8_t> from_external_tensor_proto_data;
    model_path = output_onnx;
    external_data_path = model_path.parent_path() / output_external_init_file;
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*from_external_tensor_proto, model_path, from_external_tensor_proto_data));
    size_t from_external_tensor_proto_size = from_external_tensor_proto_data.size();

    if (from_external_tensor_proto_size < model_saving_options.initializer_size_threshold) {
      // 'Small' tensors should be embedded in the onnx file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_DEFAULT, "location mismatch");
    } else {
      // 'Large' tensors should be added to the external binary file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL, "location mismatch");
    }

    ORT_RETURN_IF_NOT(tensor_proto_size == from_external_tensor_proto_size, "size mismatch");
    ORT_RETURN_IF_NOT(memcmp(tensor_proto_data.data(), from_external_tensor_proto_data.data(), tensor_proto_size) == 0, "data mismatch");

    if (model_saving_options.align_offset) {
      for (const StringStringEntryProto& entry : from_external_tensor_proto->external_data()) {
        if (entry.has_key() && entry.has_value() && entry.key() == "offset") {
          size_t tensor_offset;
          std::stringstream stream(entry.value());
          stream >> tensor_offset;
          ORT_RETURN_IF_NOT(tensor_offset % model_saving_options.on_disk_alignment == 0,
                            "tensor offset not align");
        }
      }
    }
  }
  // Cleanup.
  ORT_RETURN_IF_NOT(std::filesystem::remove(output_onnx), "delete file failed");
  ORT_RETURN_IF_NOT(std::filesystem::remove(external_data_path), "delete file failed");
  return Status::OK();
}

// Original model does not have external initializers
TEST(SaveWithExternalInitializers, Mnist) {
  ModelSavingOptions model_saving_options{100};
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/mnist.onnx"),
      ORT_TSTR(""), ORT_TSTR("testdata/mnist_with_external_initializers.onnx"),
      ORT_TSTR("mnist_external_initializers.bin"),
      model_saving_options));
}

// Original model has external initializers
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalData) {
  ModelSavingOptions model_saving_options{0};
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/model_with_orig_ext_data.onnx"),
      ORT_TSTR("model_with_orig_ext_data.onnx.data"),
      ORT_TSTR("testdata/model_with_new_external_initializers.onnx"),
      ORT_TSTR("model_with_new_external_initializers.bin"),
      model_saving_options));
}

// Original model has external initializers, align offset
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalDataAlignOffset) {
  ModelSavingOptions model_saving_options{0};
  model_saving_options.align_offset = true;
  model_saving_options.align_threshold = 0;
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(
      ORT_TSTR("testdata/model_with_orig_ext_data.onnx"),
      ORT_TSTR("model_with_orig_ext_data.onnx.data"),
      ORT_TSTR("testdata/model_with_new_external_initializers.onnx"),
      ORT_TSTR("model_with_new_external_initializers.bin"), model_saving_options));
}

namespace {

// Builds a model whose If branches own their initializers. The optimized-model
// saving path used to emit each of those twice: Node::ToProto had already placed
// them in the subgraph proto and the recursion added them again.
void AddSubgraphInitializer(GraphProto& graph, const std::string& name, int64_t rows, int64_t cols) {
  TensorProto* tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(TensorProto_DataType_FLOAT);
  tensor->add_dims(rows);
  tensor->add_dims(cols);
  tensor->set_raw_data(std::string(static_cast<size_t>(rows * cols * sizeof(float)), '\1'));
}

void MakeIfBranch(GraphProto& graph, const std::string& graph_name, const std::string& weight_name,
                  const std::string& output_name, int64_t rows, int64_t cols) {
  graph.set_name(graph_name);
  AddSubgraphInitializer(graph, weight_name, rows, cols);

  NodeProto* node = graph.add_node();
  node->set_op_type("MatMul");
  node->add_input("x");
  node->add_input(weight_name);
  node->add_output(output_name);

  ValueInfoProto* output = graph.add_output();
  output->set_name(output_name);
  auto* tensor_type = output->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_type->mutable_shape()->add_dim()->set_dim_param("N");
  tensor_type->mutable_shape()->add_dim()->set_dim_value(cols);
}

ModelProto MakeModelWithSubgraphInitializers(int64_t rows, int64_t cols) {
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  GraphProto* graph = model.mutable_graph();
  graph->set_name("main");

  ValueInfoProto* x = graph->add_input();
  x->set_name("x");
  auto* x_type = x->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(TensorProto_DataType_FLOAT);
  x_type->mutable_shape()->add_dim()->set_dim_param("N");
  x_type->mutable_shape()->add_dim()->set_dim_value(rows);

  ValueInfoProto* cond = graph->add_input();
  cond->set_name("cond");
  cond->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  cond->mutable_type()->mutable_tensor_type()->mutable_shape();

  NodeProto* if_node = graph->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("y");

  AttributeProto* then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto_AttributeType_GRAPH);
  MakeIfBranch(*then_attr->mutable_g(), "then", "Wa", "then_out", rows, cols);

  AttributeProto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto_AttributeType_GRAPH);
  MakeIfBranch(*else_attr->mutable_g(), "else", "Wb", "else_out", rows, cols);

  ValueInfoProto* y = graph->add_output();
  y->set_name("y");
  auto* y_type = y->mutable_type()->mutable_tensor_type();
  y_type->set_elem_type(TensorProto_DataType_FLOAT);
  y_type->mutable_shape()->add_dim()->set_dim_param("N");
  y_type->mutable_shape()->add_dim()->set_dim_value(cols);
  return model;
}

void CountSubgraphInitializers(const GraphProto& graph, std::map<std::string, int>& counts, int depth) {
  if (depth > 0) {
    for (const auto& tensor : graph.initializer()) {
      counts[tensor.name()]++;
    }
  }
  for (const auto& node : graph.node()) {
    for (const auto& attribute : node.attribute()) {
      if (attribute.has_g()) {
        CountSubgraphInitializers(attribute.g(), counts, depth + 1);
      }
      for (const auto& subgraph : attribute.graphs()) {
        CountSubgraphInitializers(subgraph, counts, depth + 1);
      }
    }
  }
}

// Saves a model whose subgraphs own initializers and checks that each of them is
// declared exactly once, and that the saved model can be loaded back.
void SaveAndCheckSubgraphInitializers(int64_t rows, int64_t cols, size_t size_threshold) {
  auto logger = DefaultLoggingManager().CreateLogger("SaveWithExternalInitializers");

  const auto output_onnx = std::filesystem::temp_directory_path() / "subgraph_initializers.onnx";
  const std::filesystem::path output_external{"subgraph_initializers.onnx.data"};
  std::filesystem::remove(output_onnx);
  std::filesystem::remove(output_onnx.parent_path() / output_external);

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(MakeModelWithSubgraphInitializers(rows, cols), model, nullptr, *logger));

  ModelSavingOptions saving_options{size_threshold};
  saving_options.align_offset = true;
  ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(*model, output_onnx, output_external, saving_options));

  ModelProto saved;
  {
    std::ifstream saved_stream(output_onnx, std::ios::binary);
    ASSERT_TRUE(saved_stream.is_open());
    ASSERT_TRUE(saved.ParseFromIstream(&saved_stream));
  }

  std::map<std::string, int> counts;
  CountSubgraphInitializers(saved.graph(), counts, 0);
  ASSERT_EQ(counts.size(), static_cast<size_t>(2));
  for (const auto& [name, count] : counts) {
    EXPECT_EQ(count, 1) << "subgraph initializer '" << name << "' was written " << count << " times";
  }

  std::shared_ptr<Model> reloaded;
  ASSERT_STATUS_OK(Model::Load(output_onnx.native(), reloaded, nullptr, *logger));

  std::filesystem::remove(output_onnx);
  std::filesystem::remove(output_onnx.parent_path() / output_external);
}

}  // namespace

// Subgraph initializers small enough to stay embedded in the .onnx file.
TEST(SaveWithExternalInitializers, SubgraphInitializersBelowThresholdNotDuplicated) {
  SaveAndCheckSubgraphInitializers(/*rows*/ 8, /*cols*/ 8, /*size_threshold*/ 1024);
}

// Subgraph initializers large enough to be written to the external data file.
TEST(SaveWithExternalInitializers, SubgraphInitializersAboveThresholdNotDuplicated) {
  SaveAndCheckSubgraphInitializers(/*rows*/ 512, /*cols*/ 512, /*size_threshold*/ 1024);
}

}  // namespace test
}  // namespace onnxruntime
