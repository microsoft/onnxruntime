// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"

namespace onnxruntime {
namespace nnapi {

Model::Model() : nnapi_(NnApiImplementation()) {}

Model::~Model() {
  nnapi_->ANeuralNetworksModel_free(model_);
  nnapi_->ANeuralNetworksCompilation_free(compilation_);
  nnapi_->ANeuralNetworksExecution_free(execution_);
}

}  // namespace nnapi
}  // namespace onnxruntime