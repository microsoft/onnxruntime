// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/path_string.h"
#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"

namespace onnxruntime {
namespace training {

/**
 * A checkpoint is a directory of files:
 * checkpoint/
 *   tensors.pbseq - tensor protobuf messages
 *   tensors.bin - tensor binary data
 *   properties.pbseq - property protobuf messages
 */

/**
 * Saves a model checkpoint in the specified location.
 *
 * @param checkpoint_path The checkpoint location.
 * @param data_transfer_manager The DataTransferManager instance.
 * @param runtime_tensors The tensors to persist.
 * @param properties The properties to persist.
 * @return The status of the operation.
 */
common::Status SaveModelCheckpoint(
    const PathString& checkpoint_path,
    const DataTransferManager& data_transfer_manager,
    const NameMLValMap& runtime_tensors,
    const std::unordered_map<std::string, std::string>& properties);

/**
 * Loads a model checkpoint from the specified location.
 *
 * @param checkpoint_path The checkpoint location.
 * @param model_path The model location.
 * @param tensor_protos The loaded tensors.
 * @param properties The loaded properties.
 * @return The status of the operation.
 */
common::Status LoadModelCheckpoint(
    const PathString& checkpoint_path,
    const PathString& model_path,
    std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    std::unordered_map<std::string, std::string>& properties);

}  // namespace training
}  // namespace onnxruntime
