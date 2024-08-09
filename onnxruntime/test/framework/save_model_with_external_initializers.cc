// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/data_types.h"
#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"
#include "test/test_environment.h"
#include "test_utils.h"
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
                               size_t initializer_size_threshold) {
  auto logger = DefaultLoggingManager().CreateLogger("LoadSaveAndCompareModel");
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(input_onnx, model, nullptr, *logger));
  std::filesystem::remove(output_onnx);
  std::filesystem::remove(output_external_init_file);
  ORT_RETURN_IF_ERROR(Model::SaveWithExternalInitializers(*model, output_onnx, output_external_init_file, initializer_size_threshold));

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

    std::unique_ptr<uint8_t[]> tensor_proto_data;
    size_t tensor_proto_size;
    model_path = input_onnx;
    external_data_path = (!input_external_init_file.empty()) ? (model_path.parent_path() / input_external_init_file) : std::filesystem::path();
    ORT_THROW_IF_ERROR(utils::UnpackInitializerData(*tensor_proto, external_data_path, tensor_proto_data, tensor_proto_size));

    std::unique_ptr<uint8_t[]> from_external_tensor_proto_data;
    size_t from_external_tensor_proto_size;
    model_path = output_onnx;
    external_data_path = model_path.parent_path() / output_external_init_file;
    ORT_THROW_IF_ERROR(utils::UnpackInitializerData(*from_external_tensor_proto, model_path, from_external_tensor_proto_data, from_external_tensor_proto_size));

    if (from_external_tensor_proto_size < initializer_size_threshold) {
      // 'Small' tensors should be embedded in the onnx file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_DEFAULT, "location mismatch");
    } else {
      // 'Large' tensors should be added to the external binary file.
      ORT_RETURN_IF_NOT(from_external_tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL, "location mismatch");
    }

    ORT_RETURN_IF_NOT(tensor_proto_size == from_external_tensor_proto_size, "size mismatch");
    EXPECT_EQ(memcmp(tensor_proto_data.get(), from_external_tensor_proto_data.get(), tensor_proto_size), 0);
  }
  // Cleanup.
  ORT_RETURN_IF_NOT(std::filesystem::remove(output_onnx), "delete file failed");
  ORT_RETURN_IF_NOT(std::filesystem::remove(external_data_path), "delete file failed");
  return Status::OK();
}

// Original model does not have external initializers
TEST(SaveWithExternalInitializers, Mnist) {
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(ORT_TSTR("testdata/mnist.onnx"), ORT_TSTR(""), ORT_TSTR("testdata/mnist_with_external_initializers.onnx"), ORT_TSTR("mnist_external_initializers.bin"), 100));
}

// Original model has external initializers
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalData) {
  ASSERT_STATUS_OK(LoadSaveAndCompareModel(ORT_TSTR("testdata/model_with_orig_ext_data.onnx"), ORT_TSTR("model_with_orig_ext_data.onnx.data"), ORT_TSTR("testdata/model_with_new_external_initializers.onnx"), ORT_TSTR("model_with_new_external_initializers.bin"), 0));
}

}  // namespace test
}  // namespace onnxruntime
