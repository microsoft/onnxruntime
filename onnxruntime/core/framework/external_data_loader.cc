// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader.h"
#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

common::Status IExternalDataLoader::LoadTensor([[maybe_unused]] const Env& env,
                                               [[maybe_unused]] const std::filesystem::path& data_file_path,
                                               [[maybe_unused]] FileOffsetType data_offset,
                                               [[maybe_unused]] SafeInt<size_t> data_length,
                                               [[maybe_unused]] Tensor& tensor) const {
  ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
}

}  // namespace onnxruntime
