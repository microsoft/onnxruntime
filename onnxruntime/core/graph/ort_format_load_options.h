// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

/// Options to configure how an ORT format model is loaded.
struct OrtFormatLoadOptions {
  /// If true, set initializer TensorProtos to point to memory in the flatbuffer instead of copying data.
  /// This requires the flatbuffer to remain valid for the entire duration of the InferenceSession.
  bool can_use_flatbuffer_for_initializers{true};

  /// If true, do not load any saved runtime optimizations.
  bool ignore_saved_runtime_optimizations{false};
};

}  // namespace onnxruntime
