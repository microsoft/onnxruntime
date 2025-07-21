// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_compilation_options.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/ep_context_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/environment.h"

namespace onnxruntime {
ModelCompilationOptions::ModelCompilationOptions(const onnxruntime::Environment& env, const OrtSessionOptions& session_options)
    : env_(env), session_options_(session_options) {
  session_options_.value.has_explicit_ep_context_gen_options = true;
  session_options_.value.ep_context_gen_options = session_options.value.GetEpContextGenerationOptions();
  session_options_.value.ep_context_gen_options.enable = true;
  session_options_.value.ep_context_gen_options.error_if_output_file_exists = false;

  // defaulting to kGenerateModel to support wider usage.
  session_options_.value.ep_context_gen_options.action_if_no_compiled_nodes =
      epctx::ModelGenOptions::ActionIfNoCompiledNodes::kGenerateModel;

  // Shouldn't fail because the key/value strings are below the maximum string length limits in ConfigOptions.
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1").IsOK());
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "0").IsOK());
}

void ModelCompilationOptions::SetInputModelPath(const std::filesystem::path& input_model_path) {
  ResetInputModelSettings();
  input_model_path_ = input_model_path;
}

void ModelCompilationOptions::SetInputModelFromBuffer(const void* input_model_data, size_t input_model_data_size) {
  ResetInputModelSettings();
  input_model_data_ = input_model_data;
  input_model_data_size_ = input_model_data_size;
}

Status ModelCompilationOptions::SetOutputModelPath(const std::filesystem::path& output_model_path) {
  ConfigOptions& config_options = session_options_.value.config_options;
  epctx::ModelGenOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;

  ep_context_gen_options.output_model_location = output_model_path;

  std::string output_model_path_str = PathToUTF8String(output_model_path);

  if (output_model_path_str.size() <= ConfigOptions::kMaxValueLength) {
    Status status = config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_path_str.c_str());
    ORT_ENFORCE(status.IsOK());  // Should not fail because both key/value strings are below the min string lengths
                                 // required by ConfigOptions::AddConfigEntry().
  } else {
    // A few things to note:
    //   - ORT core now uses session_options.ep_context_gen_options to read EPContext model configurations.
    //     It previously used session_options.config_options.
    //   - EPs still currently use session_options.config_options to read a subset (enabled, embed mode, output path) of
    //     EPContext model configurations.
    //     TODO(adrianlizarraga): Update EPs to use ep_context_gen_options in backward-compatible manner.
    //   - The output model file path is optional (generated from input path if absent).
    //   - EPs use the output model path to generate a path to the context binary data file IFF not embedded
    //     into EPContext nodes. If output model path is empty, EPs just create a path from input model path.
    //   - session_options.config_options limits the string length of values, which artificially limits the length
    //     of paths.
    //   - So, only add this output model file path to session_options.config_options if it is not too long. The only
    //     potential downside is that the context binary data file is using a different name, but the model will still
    //     be valid.
    logging::LoggingManager* log_manager = env_.GetLoggingManager();
    if (log_manager != nullptr && log_manager->HasDefaultLogger()) {
      const logging::Logger& logger = log_manager->DefaultLogger();
      LOGS(logger, WARNING) << "Output model path length (" << output_model_path_str.size()
                            << ") exceeds limit of " << ConfigOptions::kMaxValueLength << " characters."
                            << "ORT will still generate the expected output file, but EPs will see an empty "
                            << "output model path in SessionOption's ConfigOptions.";
    }
  }
  return Status::OK();
}

void ModelCompilationOptions::SetOutputModelExternalInitializersFile(
    const std::filesystem::path& external_initializers_path,
    size_t external_initializer_size_threshold) {
  session_options_.value.ep_context_gen_options.initializers_location = epctx::ExternalInitializerFileInfo{
      external_initializers_path,
      external_initializer_size_threshold,
  };
}

Status ModelCompilationOptions::SetOutputModelBuffer(onnxruntime::AllocatorPtr allocator,
                                                     void** output_model_buffer_ptr,
                                                     size_t* output_model_buffer_size_ptr) {
  session_options_.value.ep_context_gen_options.output_model_location = epctx::BufferHolder{
      output_model_buffer_ptr,
      output_model_buffer_size_ptr,
      std::move(allocator),
  };

  return Status::OK();
}

void ModelCompilationOptions::SetOutputModelWriteFunc(OrtWriteBufferFunc write_func, void* state) {
  session_options_.value.ep_context_gen_options.output_model_location = epctx::OutStreamHolder{
      write_func,
      state,
  };
}

void ModelCompilationOptions::SetOutputModelHandleInitializerFunc(OrtHandleInitializerDataFunc handle_initializer_func,
                                                                  void* state) {
  session_options_.value.ep_context_gen_options.initializers_location = epctx::InitializerHandler{
      handle_initializer_func,
      state,
  };
}

Status ModelCompilationOptions::SetEpContextBinaryInformation(const std::filesystem::path& output_directory,
                                                              const std::filesystem::path& model_name) {
  if (output_directory.empty() || model_name.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "output_dir or model_name is empty.");
  }

  if (output_directory.has_filename() && output_directory.extension() == "") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "output_dir is not a valid directory.");
  }

  std::filesystem::path ctx_model_path = output_directory / model_name;
  std::string ctx_model_path_str = PathToUTF8String(ctx_model_path);

  if (ctx_model_path_str.size() <= ConfigOptions::kMaxValueLength) {
    ORT_RETURN_IF_ERROR(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                                                                             ctx_model_path_str.c_str()));
  } else {
    logging::LoggingManager* log_manager = env_.GetLoggingManager();
    if (log_manager != nullptr && log_manager->HasDefaultLogger()) {
      const logging::Logger& logger = log_manager->DefaultLogger();
      LOGS(logger, WARNING) << "output_directory length with model_name length together exceeds limit of "
                            << ConfigOptions::kMaxValueLength << " characters."
                            << "ORT will still generate the expected output file, but EPs will see an empty "
                            << "output path in SessionOption's ConfigOptions.";
    }
  }

  session_options_.value.ep_context_gen_options.output_model_path_hint = ctx_model_path;

  return Status::OK();
}

Status ModelCompilationOptions::SetEpContextEmbedMode(bool embed_ep_context_in_model) {
  ORT_RETURN_IF_ERROR(session_options_.value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextEmbedMode, embed_ep_context_in_model ? "1" : "0"));
  session_options_.value.ep_context_gen_options.embed_ep_context_in_model = embed_ep_context_in_model;
  return Status::OK();
}

void ModelCompilationOptions::SetEpContextDataWriteFunc(OrtWriteEpContextDataFunc write_func, void* state) {
  session_options_.value.ep_context_gen_options.write_ep_context_data_func = write_func;
  session_options_.value.ep_context_gen_options.write_ep_context_data_state = state;
}

Status ModelCompilationOptions::SetFlags(size_t flags) {
  epctx::ModelGenOptions& options = session_options_.value.ep_context_gen_options;
  options.error_if_output_file_exists = flags & OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS;
  options.action_if_no_compiled_nodes =
      (flags & OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED) ? epctx::ModelGenOptions::ActionIfNoCompiledNodes::kReturnError
                                                              : epctx::ModelGenOptions::ActionIfNoCompiledNodes::kGenerateModel;
  return Status::OK();
}

const OrtSessionOptions& ModelCompilationOptions::GetSessionOptions() const {
  return session_options_;
}

bool ModelCompilationOptions::InputModelComesFromFile() const {
  return !input_model_path_.empty();
}

const std::filesystem::path& ModelCompilationOptions::GetInputModelPath() const {
  return input_model_path_;
}

const void* ModelCompilationOptions::GetInputModelData() const {
  return input_model_data_;
}

size_t ModelCompilationOptions::GetInputModelDataSize() const {
  return input_model_data_size_;
}

void ModelCompilationOptions::ResetInputModelSettings() {
  input_model_path_.clear();
  input_model_data_ = nullptr;
  input_model_data_size_ = 0;
}

Status ModelCompilationOptions::Check() const {
  const ConfigOptions& config_options = session_options_.value.config_options;

  ORT_ENFORCE(session_options_.value.ep_context_gen_options.enable);
  ORT_ENFORCE(config_options.GetConfigOrDefault(kOrtSessionOptionsDisableModelCompile, "0") == "0");

  // Check input model settings.
  const bool input_from_file = !input_model_path_.empty();
  const bool input_from_memory = input_model_data_ != nullptr;

  if (!input_from_file && !input_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer");
  }

  if (input_from_file && input_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer, ",
                           "but not both.");
  }

  if (input_from_file && !std::filesystem::exists(input_model_path_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input model path does not exist: ", input_model_path_);
  }

  if (input_from_memory && input_model_data_size_ == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer for input model data has size 0");
  }

  // Check output model settings.
  const epctx::ModelGenOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;
  bool has_no_output_model_location = std::holds_alternative<std::monostate>(
      ep_context_gen_options.output_model_location);

  if (has_no_output_model_location && input_from_file) {
    // User did not specify an output file, an output buffer, or an output write function. We default to generating an
    // output file with a name based on the input file name, so do not return an error.
    return Status::OK();
  }

  if (has_no_output_model_location) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to generate an output model path: require an input model path if the location "
                           "of the output model (e.g., file, buffer, or stream) is not specified.");
  }

  const epctx::BufferHolder* output_buffer_ptr = ep_context_gen_options.TryGetOutputModelBuffer();

  if (output_buffer_ptr != nullptr && output_buffer_ptr->buffer_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: buffer pointer is null");
  }

  if (output_buffer_ptr != nullptr && output_buffer_ptr->buffer_size_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: size pointer is null");
  }

  if (output_buffer_ptr != nullptr && output_buffer_ptr->buffer_allocator == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: allocator is null");
  }

  const epctx::OutStreamHolder* output_stream_ptr = ep_context_gen_options.TryGetOutputModelOutStream();

  if (output_stream_ptr != nullptr && output_stream_ptr->write_func == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid write-to-stream function for output model: function pointer is null");
  }

  if (ep_context_gen_options.write_ep_context_data_func != nullptr &&
      ep_context_gen_options.embed_ep_context_in_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "EpContextEmbedMode must be false to use a function that writes out EPContext ",
                           "node binary data (i.e., OrtEpContextDataWriteFunc).");
  }

  return Status::OK();
}

}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
