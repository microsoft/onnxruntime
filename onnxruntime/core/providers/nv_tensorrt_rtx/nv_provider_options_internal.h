#ifndef ONNXRUNTIME_NV_PROVIDER_OPTIONS_INTERNAL_H
#define ONNXRUNTIME_NV_PROVIDER_OPTIONS_INTERNAL_H

#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"  // Include the public header first

// Include necessary standard headers
#include <stddef.h>   // For size_t
#include <stdint.h>   // For fixed-width types like uint32_t
#include <stdbool.h>  // For bool type

// Define a magic number to help verify the handle validity at runtime
#define ORT_NV_PROVIDER_OPTIONS_MAGIC 0x4E56504F  // ASCII for "NVPO"

/**
 * \brief Internal structure definition for OrtNvTensorRtRtxProviderOptions.
 *
 * This definition is hidden from the public API users.
 * It contains the actual configuration data for the Nv provider.
 */
struct OrtNvTensorRtRtxProviderOptions {
  // Magic number for basic type validation in API calls
  uint32_t magic_number;

  // --- Configuration Members ---

  int device_id;                 // CUDA device ID.
  bool has_user_compute_stream;  // Indicator if user specified a CUDA compute stream. (Internal flag derived from user_compute_stream != NULL)
  void* user_compute_stream;     // User specified CUDA compute stream. NULL if not set.
  bool cuda_graph_enable;        // Enable CUDA graph capture.
  size_t max_workspace_size;     // Maximum workspace size in bytes. 0 means implementation defined default.
  bool dump_subgraphs;           // Dump EP subgraphs.
  bool detailed_build_log;       // Enable detailed TensorRT engine build logging.

  // --- TensorRT Profile Shapes  ---
  const char* profile_min_shapes;  // Min shapes string.
  const char* profile_max_shapes;  // Max shapes string.
  const char* profile_opt_shapes;  // Optimal shapes string.

  // --- TensorRT ONNX Byte Stream for Weights ---
  // This pointer points to memory owned by the USER.
  const void* onnx_bytestream;  // Pointer to original ONNX model byte stream.
  size_t onnx_bytestream_size;  // Size of the byte stream.
};

#endif  // ONNXRUNTIME_NV_PROVIDER_OPTIONS_INTERNAL_H
