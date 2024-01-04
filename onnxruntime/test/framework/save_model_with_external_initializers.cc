// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

void LoadSaveAndCompareModel(const std::string& input_onnx,
                             const std::string& input_external_init_file,
                             const std::string& output_onnx,
                             const std::string& output_external_init_file,
                             size_t initializer_size_threshold) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(input_onnx), model, nullptr, DefaultLoggingManager().DefaultLogger()));
  std::remove(output_onnx.c_str());
  std::remove(output_external_init_file.c_str());
  ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(*model, ToPathString(output_onnx), output_external_init_file, initializer_size_threshold));

  std::shared_ptr<Model> model_from_external;
  ASSERT_STATUS_OK(Model::Load(ToPathString(output_onnx), model_from_external, nullptr, DefaultLoggingManager().DefaultLogger()));

  Graph& graph = model->MainGraph();
  // Perform shape inference on the graph, if this succeeds then it means that we could correctly read the
  // integer initializers used by reshape and transpose.
  ASSERT_STATUS_OK(graph.Resolve());
  Graph& graph_from_external = model_from_external->MainGraph();

  InitializedTensorSet initializers = graph.GetAllInitializedTensors();
  InitializedTensorSet initializers_from_external = graph_from_external.GetAllInitializedTensors();

  ASSERT_EQ(initializers.size(), initializers_from_external.size());

  // Compare the initializers of the two versions.
  Path model_path{};
  Path external_data_path{};
  for (auto i : initializers) {
    const std::string kInitName = i.first;
    const ONNX_NAMESPACE::TensorProto* tensor_proto = i.second;
    const ONNX_NAMESPACE::TensorProto* from_external_tensor_proto = initializers_from_external[kInitName];

    std::vector<uint8_t> tensor_proto_data;
    model_path = Path::Parse(ToPathString(input_onnx));
    external_data_path = (input_external_init_file.size()) ? model_path.ParentPath().Append(Path::Parse(ToPathString(input_external_init_file))) : Path();
    ORT_THROW_IF_ERROR(utils::UnpackInitializerData(*tensor_proto, external_data_path, tensor_proto_data));
    size_t tensor_proto_size = tensor_proto_data.size();

    std::vector<uint8_t> from_external_tensor_proto_data;
    model_path = Path::Parse(ToPathString(output_onnx));
    external_data_path = model_path.ParentPath().Append(Path::Parse(ToPathString(output_external_init_file)));
    ORT_THROW_IF_ERROR(utils::UnpackInitializerData(*from_external_tensor_proto, model_path, from_external_tensor_proto_data));
    size_t from_external_tensor_proto_size = from_external_tensor_proto_data.size();

    if (from_external_tensor_proto_size < initializer_size_threshold) {
      // 'Small' tensors should be embedded in the onnx file.
      EXPECT_EQ(from_external_tensor_proto->data_location(), ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_DEFAULT);
    } else {
      // 'Large' tensors should be added to the external binary file.
      EXPECT_EQ(from_external_tensor_proto->data_location(), ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
    }

    ASSERT_EQ(tensor_proto_size, from_external_tensor_proto_size);
    EXPECT_EQ(memcmp(tensor_proto_data.data(), from_external_tensor_proto_data.data(), tensor_proto_size), 0);
  }
  // Cleanup.
  ASSERT_EQ(std::remove(output_onnx.c_str()), 0);
  ASSERT_EQ(std::remove(PathToUTF8String(external_data_path.ToPathString()).c_str()), 0);
}

// Original model does not have external initializers
TEST(SaveWithExternalInitializers, Mnist) {
  LoadSaveAndCompareModel("testdata/mnist.onnx", "", "testdata/mnist_with_external_initializers.onnx", "mnist_external_initializers.bin", 100);
}

// Original model has external initializers
TEST(SaveWithExternalInitializers, ModelWithOriginalExternalData) {
  LoadSaveAndCompareModel("testdata/model_with_orig_ext_data.onnx", "model_with_orig_ext_data.onnx.data", "testdata/model_with_new_external_initializers.onnx", "model_with_new_external_initializers.bin", 0);
}

}  // namespace test
}  // namespace onnxruntime
