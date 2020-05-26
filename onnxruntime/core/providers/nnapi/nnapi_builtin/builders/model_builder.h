// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/onnx_pb.h>
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "Shaper.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"

namespace onnxruntime {
namespace nnapi {

class ModelBuilder {
 private:
  const NnApi* nnapi_{nullptr};
  ONNX_NAMESPACE::ModelProto& model_proto_;
  std::pair<bool, std::string> IsNodeSupported(const ONNX_NAMESPACE::NodeProto& node);
  Shaper shaper_;

 public:
  ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto, const NnApi* nnapi);
  ~ModelBuilder() = default;
  std::vector<std::vector<int>> GetSupportedNodes();
  std::unique_ptr<Model> Compile();
};

}  // namespace nnapi
}  // namespace onnxruntime