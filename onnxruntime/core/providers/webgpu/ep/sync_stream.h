// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class WebGpuExecutionProvider;
namespace webgpu {
struct CommandRecordingState;
namespace ep {

OrtSyncStreamImpl* CreateWebGpuSyncStream(WebGpuExecutionProvider& ep);
CommandRecordingState& GetWebGpuStreamCommandState(const OrtSyncStream* stream);
common::Status CopyTensorOnWebGpuStream(const OrtSyncStream* stream, const void* src_data,
                                        bool src_is_gpu, void* dst_data, bool dst_is_gpu, size_t bytes);

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
