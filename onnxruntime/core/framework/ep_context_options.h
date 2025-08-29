// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <filesystem>
#include <streambuf>
#include <string>
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
struct OutStreamHolder {
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
  OrtHandleInitializerDataFunc handle_initializer_func = nullptr;
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

  ModelGenOptions() = default;

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
               OutStreamHolder>        // Function to write the output model to a user's stream.
      output_model_location{};

  std::variant<std::monostate,               // Initial state (initializers embedded in ONNX model).
               ExternalInitializerFileInfo,  // Initializers saved in an external file
               InitializerHandler>           // Custom function called for every initializer to determine location.
      initializers_location{};

  bool HasOutputModelLocation() const;
  const std::filesystem::path* TryGetOutputModelPath() const;
  const BufferHolder* TryGetOutputModelBuffer() const;
  const OutStreamHolder* TryGetOutputModelOutStream() const;

  bool AreInitializersEmbeddedInOutputModel() const;
  const ExternalInitializerFileInfo* TryGetExternalInitializerFileInfo() const;
  const InitializerHandler* TryGetInitializerHandler() const;
};

#if !defined(ORT_MINIMAL_BUILD)
// Class that wraps the user's OrtOutStreamWriteFunc function to enable use with
// C++'s std::ostream.
// Example:
//    OutStreamHolder stream_holder{write_func, stream_state};
//    std::unique_ptr<OutStreamBuf> out_stream_buf = std::make_unique<OutStreamBuf>(stream_holder);
//    std::ostream out_stream(out_stream_buf.get());
class OutStreamBuf : public std::streambuf {
 public:
  explicit OutStreamBuf(OutStreamHolder out_stream_holder);
  ~OutStreamBuf();

  const Status& GetStatus() const {
    return last_status_;
  }

 protected:
  int_type overflow(int_type ch) override;
  int sync() override;

 private:
  OutStreamHolder out_stream_holder_{};
  std::array<char, 4096> buffer_{};
  Status last_status_{};
};
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace epctx
}  // namespace onnxruntime
