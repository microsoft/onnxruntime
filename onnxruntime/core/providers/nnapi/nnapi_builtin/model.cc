// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "nnapi_lib/nnapi_implementation.h"

namespace onnxruntime {
namespace nnapi{

Model::Model() : nnapi_(NnApiImplementation()) {}

} } // namespace onnxruntime::nnapi