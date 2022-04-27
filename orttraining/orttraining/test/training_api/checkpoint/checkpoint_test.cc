// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpointing.h"

#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/path_lib.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/utilities.h"
#include "orttraining/training_api/interfaces.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "test/test_environment.h"
#include "test/util/include/test/test_environment.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"

using onnxruntime::test::TemporaryDirectory;

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {
#define MODEL_FOLDER ORT_TSTR("testdata/")

TEST(CheckPointApiTest, Save_CPU) {
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/e2e.onnx";
  std::shared_ptr<Model> p_model;

  const auto& default_logger = onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, default_logger));
  Graph& graph = p_model->MainGraph();

  std::vector<std::string> trainable_weight_names{
      "bert.encoder.layer.2.output.LayerNorm.weight",
      "bert.encoder.layer.2.output.LayerNorm.bias",
      "add1_initializerr",
      "cls.predictions.transform.LayerNorm.weight",
      "cls.predictions.transform.LayerNorm.bias",
      "bert.embeddings.word_embeddings.weight_transposed",
      "cls.predictions.bias",
  };

  std::vector<const ONNX_NAMESPACE::TensorProto*> trainable_weight_values;
  trainable_weight_values.reserve(trainable_weight_names.size());
  for (size_t i = 0; i < trainable_weight_names.size(); ++i) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    ORT_ENFORCE(graph.GetInitializedTensor(trainable_weight_names[i], tensor_proto), "Failed to find weight values");
    trainable_weight_values.emplace_back(tensor_proto);
  }

  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }

  TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ORT_ENFORCE(api_test::CheckpointUtils::SaveORTCheckpoint(trainable_weight_values, trainable_weight_names, checkpoint_path).IsOK());

  // Check the ckpt files in the directory.
  // std::unordered_map<std::string, PathString> group_folder_paths;
  // LoopDir(checkpoint_path,
  //         [&group_folder_paths, &checkpoint_path](const PathChar* filename, OrtFileType file_type) -> bool {
  //           PathString filename_str = filename;
  //           if (filename_str[0] == '.' ||
  //               file_type != OrtFileType::TYPE_DIR) {
  //             return true;
  //           }
  //           group_folder_paths.insert({filename_str, ConcatPathComponent<PathChar>(optimizer_folder_path, filename_str)});
  //           return true;
  //         });
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime