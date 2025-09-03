// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <variant>
#include "core/framework/allocator.h"
#include "core/framework/config_options.h"

namespace onnxruntime {
namespace epctx {
/// <summary>
/// Holds the buffer that will store the output model and the allocator used to allocate the memory.
/// </summary>
struct BufferHolder {
  void** buffer_ptr = nullptr;
  size_t* buffer_size_ptr = nullptr;
  AllocatorPtr buffer_allocator = nullptr;
};

/// <summary>
/// Holds the opaque stream state and the write function that ORT calls to write out the output model.
/// </summary>
struct BufferWriteFuncHolder {
  OrtWriteBufferFunc write_func = nullptr;
  void* stream_state = nullptr;  // Opaque pointer to user's stream state. Passed as first argument to write_func.
};

/// <summary>
/// Holds path and size threshold used to write out initializers to an external file.
/// </summary>
struct ExternalInitializerFileInfo {
  std::filesystem::path file_path;
  size_t size_threshold = 0;
};

/// <summary>
/// Holds function and state provided by user to handle initializer data (i.e., write to stream or embed in model).
/// </summary>
struct InitializerHandler {
  OrtGetInitializerLocationFunc handle_initializer_func = nullptr;
  void* state = nullptr;
};

/// <summary>
/// Stores EPContext model generation options. Used in SessionOptions.
/// </summary>
struct ModelGenOptions {
  // Action to take if the output model does not have compiled (EPContext) nodes.
  enum class ActionIfNoCompiledNodes {
    // Return OK() but don't generate an output model. Compiling via SessionOptions defaults to this behavior
    // to maintain compatibility. The explicit compile API does *not* use this action.
    kDontGenerateModel = 0,

    // Generate an output model even if it doesn't have compiled nodes.
    // The explicit Compile API defaults to this value.
    kGenerateModel,

    // Return an error if the model does not have compiled nodes.
    // The explicit Compile API can be configured to this value.
    kReturnError,
  };

  ModelGenOptions();

  // Initializes from string key/value pairs in session config options.
  explicit ModelGenOptions(const ConfigOptions& config_options);

  bool enable = false;
  bool error_if_output_file_exists = true;
  bool error_if_no_compiled_nodes = false;
  bool embed_ep_context_in_model = false;
  ActionIfNoCompiledNodes action_if_no_compiled_nodes = ActionIfNoCompiledNodes::kDontGenerateModel;

  std::variant<std::monostate,         // Initial state (no output model location)
               std::filesystem::path,  // output model path
               BufferHolder,           // buffer to save output model
               BufferWriteFuncHolder>  // Function to write the output model to a user's stream.
      output_model_location = std::monostate{};

  std::variant<std::monostate,               // Initial state (initializers embedded in ONNX model).
               ExternalInitializerFileInfo,  // Initializers saved to a single external file depending on size.
               InitializerHandler>           // Custom function called for every initializer to determine location.
      initializers_location = std::monostate{};

  bool HasOutputModelLocation() const;
  const std::filesystem::path* TryGetOutputModelPath() const;
  const BufferHolder* TryGetOutputModelBuffer() const;
  const BufferWriteFuncHolder* TryGetOutputModelWriteFunc() const;

  bool AreInitializersEmbeddedInOutputModel() const;
  const ExternalInitializerFileInfo* TryGetExternalInitializerFileInfo() const;
  const InitializerHandler* TryGetInitializerHandler() const;
};

}  // namespace epctx
}  // namespace onnxruntime
