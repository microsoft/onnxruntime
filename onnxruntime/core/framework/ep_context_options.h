// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <variant>
#include "core/framework/allocator.h"
#include "core/framework/config_options.h"

namespace onnxruntime {
namespace epctx {
struct BufferHolder {
  void** buffer_ptr = nullptr;
  size_t* buffer_size_ptr = nullptr;
  AllocatorPtr buffer_allocator = nullptr;
};

struct StreamHolder {
  WriteToStreamFunc write_func;
  void* state;  // Opaque pointer to user's stream state. Passed as first argument to write_func.
};

/*
class WriteFuncStreamBuf : public std::streambuf {
 public:
  WriteFuncStreamBuf(StreamHolder write_func_holder);
  ~WriteFuncStreamBuf();

 protected:
  int_type overflow(int_type ch) override;
  int sync() override;

 private:
  int FlushBuffer();

  StreamHolder write_func_holder_;
  std::array<char, 4096> buffer_;
};
*/

struct ModelGenOptions {
  ModelGenOptions() = default;

  // Initializes from string key/value pairs in session config options.
  explicit ModelGenOptions(const ConfigOptions& config_options);

  bool enable = false;
  bool overwrite_existing_output_file = false;
  bool error_if_no_compiled_nodes = false;
  bool embed_ep_context_in_model = false;

  std::variant<std::monostate,  // Initial state (no output model location)
               std::string,     // output model path
               BufferHolder,    // buffer to save output model
               StreamHolder>    // Function to write the output model to a user's stream.
      output_model_location;

  std::string output_external_initializers_file_path;
  size_t output_external_initializer_size_threshold = 0;

  bool HasOutputModelLocation() const;
  const std::string* TryGetOutputModelPath() const;
  const BufferHolder* TryGetOutputModelBuffer() const;
  const StreamHolder* TryGetOutputModelStream() const;
};
}  // namespace epctx
}  // namespace onnxruntime
