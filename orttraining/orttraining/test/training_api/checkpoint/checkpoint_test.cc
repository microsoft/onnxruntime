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

TEST(CheckPointApiTest, SaveOnnxModelAsCheckpoint_CPU) {
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/e2e.onnx";
  std::vector<std::string> trainable_weight_names{
      "bert.encoder.layer.2.output.LayerNorm.weight",
      "bert.encoder.layer.2.output.LayerNorm.bias",
      "add1_initializerr",
      "cls.predictions.transform.LayerNorm.weight",
      "cls.predictions.transform.LayerNorm.bias",
      "bert.embeddings.word_embeddings.weight_transposed",
      "cls.predictions.bias",
  };

  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }

  TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ORT_ENFORCE(api_test::CheckpointUtils::SaveORTCheckpoint(model_uri, trainable_weight_names, checkpoint_path).IsOK());

  // Check the ckpt files in the directory.
  std::set<PathString> expected_file_names{k_tensors_file_name};
  std::set<PathString> valid_file_names;
  LoopDir(checkpoint_path,
          [&valid_file_names, &checkpoint_path](const PathChar* filename, OrtFileType file_type) -> bool {
            PathString filename_str = filename;
            bool is_valid_ckpt_file_exts =
                HasExtensionOf(filename_str, ORT_TSTR("pbseq")) || HasExtensionOf(filename_str, ORT_TSTR("bin"));
            if (filename_str[0] == '.' || file_type == OrtFileType::TYPE_DIR || !is_valid_ckpt_file_exts) {
              return true;
            }
            valid_file_names.emplace(filename_str);
            return true;
          });

  ASSERT_EQ(expected_file_names, valid_file_names);
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime