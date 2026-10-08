// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class WebGpuExecutionProvider;
namespace webgpu {
struct CommandRecordingState;
namespace ep {

OrtSyncStreamImpl* CreateWebGpuSyncStream(WebGpuExecutionProvider& ep);
constexpr bool ShouldUseLegacyRecording(uint32_t ort_api_version, uint32_t ort_patch_version, bool force_legacy) {
  // Patch-level fixes in the 1.28 and 1.30 release lines do not imply support in 1.29.
  const bool supports_session_recording = ort_api_version >= 31 ||
                                          (ort_api_version == 30 && ort_patch_version >= 1) ||
                                          (ort_api_version == 28 && ort_patch_version >= 3);
  return force_legacy || !supports_session_recording;
}

bool UseLegacyRecording();
CommandRecordingState& GetWebGpuStreamCommandState(const OrtSyncStream* stream);
common::Status CopyTensorOnWebGpuStream(const OrtSyncStream* stream, const void* src_data,
                                        bool src_is_gpu, void* dst_data, bool dst_is_gpu, size_t bytes);

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
