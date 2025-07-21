// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
class Environment;

/// <summary>
/// Stores options to compile ONNX models into "EPContext" models.
/// </summary>
class ModelCompilationOptions {
 public:
  /// <summary>
  /// Creates an instance with the session options to use for model compilation.
  /// The session options are expected to have execution providers that compile.
  /// </summary>
  /// <param name="env">Reference to Environment</param>
  /// <param name="session_options">Reference to session options</param>
  ModelCompilationOptions(const onnxruntime::Environment& env, const OrtSessionOptions& session_options);

  /// <summary>
  /// Sets the file path to the input ONNX model to compile.
  /// Overrides any previous call to SetInputModelPath() or SetInputModelFromBuffer().
  /// </summary>
  /// <param name="input_model_path">The input model's path</param>
  void SetInputModelPath(const std::string& input_model_path);

  /// <summary>
  /// Sets the buffer that stores the input ONNX model to compile.
  /// Overrides any previous call to SetInputModelPath() or SetInputModelFromBuffer().
  /// </summary>
  /// <param name="input_model_data">Buffer containing the input ONNX model</param>
  /// <param name="input_model_data_size">The size in bytes of the input model's buffer</param>
  void SetInputModelFromBuffer(const void* input_model_data, size_t input_model_data_size);

  /// <summary>
  /// Sets the file path to store the output/compiled ONNX model.
  /// Overrides any previous call to SetOutputModelPath() or SetOutputModelBuffer().
  /// </summary>
  /// <param name="output_model_path"></param>
  /// <returns>Status indicating potential error</returns>
  Status SetOutputModelPath(const std::string& output_model_path);

  /// <summary>
  /// Sets the file path to the file that will store external ONNX initializers for the compiled model.
  /// Only stores initializers for graph nodes assigned to CPU EP.
  /// </summary>
  /// <param name="external_initializers_path">Path to the external initializers file to generate</param>
  /// <param name="external_initializer_size_threshold">Initializers that exceed this threshold are external</param>
  void SetOutputModelExternalInitializersFile(const std::string& external_initializers_path,
                                              size_t external_initializer_size_threshold);

  /// <summary>
  /// Sets a pointer to the buffer that will contained the output/compiled ONNX model bytes.
  /// Overrides any previous call to SetOutputModelPath() or SetOutputModelBuffer().
  /// </summary>
  /// <param name="allocator">Allocator to allocate the output buffer</param>
  /// <param name="output_model_buffer_ptr">Pointer to the buffer that will contain the compiled model</param>
  /// <param name="output_model_buffer_size_ptr">Set to the size of the buffer</param>
  /// <returns>Status indicating potential error</returns>
  Status SetOutputModelBuffer(onnxruntime::AllocatorPtr allocator, void** output_model_buffer_ptr,
                              size_t* output_model_buffer_size_ptr);

  /// <summary>
  /// Sets an output stream (write function + state) used to write out the compiled model bytes.
  /// </summary>
  /// <param name="write_func">Write function</param>
  /// <param name="state">The user's state</param>
  void SetOutputModelWriteFunc(OrtWriteBufferFunc write_func, void* state);

  /// <summary>
  /// Sets a user-provided function to handle serialization of ONNX initializers.
  /// </summary>
  /// <param name="handle_initializer_func">The user-provided function called for every initializer</param>
  /// <param name="state">The user's state.</param>
  void SetOutputModelHandleInitializerFunc(OrtHandleInitializerDataFunc handle_initializer_func, void* state);

  /// <summary>
  /// Sets information relate to EP context binary file.
  /// EP use this information to decide the location and context binary file name.
  /// Used while compiling model with input and output in memory buffer
  /// </summary>
  /// <param name="output_directory">The folder path to the generated context binary file</param>
  /// <param name="model_name">Model name used to decide the context binary file name: [model_name]_[ep].bin</param>
  /// <returns>Status indicating potential error</returns>
  Status SetEpContextBinaryInformation(const std::filesystem::path& output_directory,
                                       const std::filesystem::path& model_name);

  /// <summary>
  /// Enables or disables the embedding of EPContext binary data into the `ep_cache_context` attribute of EPContext
  /// nodes. Defaults to false (dumped to file).
  /// </summary>
  /// <param name="embed_ep_context_in_model">True if should be embedded, false otherwise</param>
  /// <returns>Status indicating potential error</returns>
  Status SetEpContextEmbedMode(bool embed_ep_context_in_model);

  /// <summary>
  /// Sets an output stream (write function + state) used to write out EPContext node binary data.
  /// </summary>
  /// <param name="write_func">Write function</param>
  /// <param name="state">The user's state</param>
  void SetEpContextDataWriteFunc(OrtWriteEpContextDataFunc write_func, void* state);

  /// <summary>
  /// Sets flags representing enabled boolean options defined in OrtCompileApiFlags.
  /// </summary>
  /// <param name="flags">unsigned integer set to the bitwise OR of enabled flags.</param>
  /// <returns>Status indicating success or an error</returns>
  Status SetFlags(size_t flags);

  /// <summary>
  /// Returns a reference to the session options object.
  /// </summary>
  /// <returns>session options</returns>
  const OrtSessionOptions& GetSessionOptions() const;

  /// <summary>
  /// Returns the file path to the input ONNX model.
  /// </summary>
  /// <returns>input model's path</returns>
  const std::string& GetInputModelPath() const;

  /// <summary>
  /// Returns true if the input model is read from a file.
  /// </summary>
  /// <returns>true if input model comes from a file</returns>
  bool InputModelComesFromFile() const;

  /// <summary>
  /// Returns the buffer that contains the bytes for the input ONNX model.
  /// Returns nullptr if the input model is not stored in a buffer.
  /// </summary>
  /// <returns>pointer to input model's buffer</returns>
  const void* GetInputModelData() const;

  /// <summary>
  /// Returns the size in bytes of the buffer that contains the input ONNX model.
  /// Returns 0 if the input model is not stored in a buffer.
  /// </summary>
  /// <returns>input model buffer's size in bytes</returns>
  size_t GetInputModelDataSize() const;

  /// <summary>
  /// Checks if the compilation options described by this object are valid.
  /// </summary>
  /// <returns>An error status if the compilation options are invalid</returns>
  Status Check() const;

 private:
  void ResetInputModelSettings();
  Status CheckInputModelSettings() const;
  Status CheckOutputModelSettings() const;

  const onnxruntime::Environment& env_;
  OrtSessionOptions session_options_;
  std::string input_model_path_;
  const void* input_model_data_ = nullptr;
  size_t input_model_data_size_ = 0;
};
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
