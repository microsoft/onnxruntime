// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class PrepackedWeightsForGraph;

// These options affect how the model initializers are written to the external file.
// This includes options to align external initializer offset.
// ORT will try to use mmap to load external initializers.
//
// ORT saves external initializers into single data file, each initializer is
// accessed with offset(start position of initializer) and length(byte length of
// initializer) of the data file. With align_offset to true, ORT will align
// offset for large initializer (larger than  align_threshold)
// when save ONNX model with external data file. It will align then to
// on_disk_alignment value.
struct ModelSavingOptions {
  explicit ModelSavingOptions(size_t size_threshold)
      : initializer_size_threshold(size_threshold) {}

  // Minimal initializer size in bytes to be externalized on disk
  size_t initializer_size_threshold;
  // Offset will always be aligned for mmap support.
  // This is done by padding previous tensor data with zeros keeping same length.
  bool align_offset = false;
  // Alignment threshold for size of data.
  // Having a low threshold will waste file space for small initializers.
  // Only when tensor's data size is > the page_align_threshold it will be force
  // aligned. Default to 1MB.
  int64_t align_threshold = 1048576;
  // Alignment factor for big tensors (bigger than align_threshold). Defaults to 4K.
  int64_t on_disk_alignment = 4096;
  // Force embed all external initializer into the Onnx file
  // Used for EPContext model generation while some nodes fallback on CPU which has external data dependency
  bool force_embed_external_ini = false;
};

}  // namespace onnxruntime
