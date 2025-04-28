// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_compilation_options.h"

#include <memory>
#include <string>
#include <utility>

#include "core/session/allocator_adapters.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_env.h"

namespace onnxruntime {
ModelCompilationOptions::ModelCompilationOptions(const OrtEnv& env, const OrtSessionOptions& session_options)
    : env_(env), session_options_(session_options) {
  session_options_.value.has_explicit_ep_context_gen_options = true;
  session_options_.value.ep_context_gen_options = session_options.value.GetEpContextGenerationOptions();
  session_options_.value.ep_context_gen_options.enable = true;
  session_options_.value.ep_context_gen_options.overwrite_existing_output_file = true;
  session_options_.value.ep_context_gen_options.error_if_no_compiled_nodes = true;

  // Shouldn't fail because the key/value strings are below the maximum string length limits in ConfigOptions.
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1").IsOK());
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "0").IsOK());
}

void ModelCompilationOptions::SetInputModelPath(const std::string& input_model_path) {
  ResetInputModelSettings();
  input_model_path_ = input_model_path;
}

void ModelCompilationOptions::SetInputModelFromBuffer(const void* input_model_data, size_t input_model_data_size) {
  ResetInputModelSettings();
  input_model_data_ = input_model_data;
  input_model_data_size_ = input_model_data_size;
}

Status ModelCompilationOptions::SetOutputModelPath(const std::string& output_model_path) {
  ORT_RETURN_IF_ERROR(ResetOutputModelSettings());

  ConfigOptions& config_options = session_options_.value.config_options;
  EpContextModelGenerationOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;

  ep_context_gen_options.output_model_file_path = output_model_path;

  if (ep_context_gen_options.output_model_file_path.size() <= ConfigOptions::kMaxValueLength) {
    Status status = config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                                                  ep_context_gen_options.output_model_file_path.c_str());
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
      LOGS(logger, WARNING) << "Output model path length (" << ep_context_gen_options.output_model_file_path.size()
                            << ") exceeds limit of " << ConfigOptions::kMaxKeyLength << " characters."
                            << "ORT will still generated the expected output file, but EPs will see an empty "
                            << "output model path in SessionOption's ConfigOptions.";
    }
  }
  return Status::OK();
}

void ModelCompilationOptions::SetOutputModelExternalInitializersFile(const std::string& external_initializers_path,
                                                                     size_t external_initializer_size_threshold) {
  session_options_.value.ep_context_gen_options.output_external_initializers_file_path = external_initializers_path;
  session_options_.value.ep_context_gen_options.output_external_initializer_size_threshold =
      external_initializer_size_threshold;
}

Status ModelCompilationOptions::SetOutputModelBuffer(OrtAllocator* allocator,
                                                     void** output_model_buffer_ptr,
                                                     size_t* output_model_buffer_size_ptr) {
  ORT_RETURN_IF_ERROR(ResetOutputModelSettings());

  session_options_.value.ep_context_gen_options.output_model_buffer_ptr = output_model_buffer_ptr;
  session_options_.value.ep_context_gen_options.output_model_buffer_size_ptr = output_model_buffer_size_ptr;
  session_options_.value.ep_context_gen_options.output_model_buffer_allocator =
      std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  return Status::OK();
}

Status ModelCompilationOptions::SetEpContextEmbedMode(bool embed_ep_context_in_model) {
  ORT_RETURN_IF_ERROR(session_options_.value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextEmbedMode, embed_ep_context_in_model ? "1" : "0"));
  session_options_.value.ep_context_gen_options.embed_ep_context_in_model = embed_ep_context_in_model;
  return Status::OK();
}

const OrtSessionOptions& ModelCompilationOptions::GetSessionOptions() const {
  return session_options_;
}

bool ModelCompilationOptions::InputModelComesFromFile() const {
  return !input_model_path_.empty();
}

const std::string& ModelCompilationOptions::GetInputModelPath() const {
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

Status ModelCompilationOptions::ResetOutputModelSettings() {
  EpContextModelGenerationOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;
  ep_context_gen_options.output_model_file_path.clear();
  ep_context_gen_options.output_model_buffer_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_size_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_allocator = nullptr;
  return session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");
}

Status ModelCompilationOptions::CheckInputModelSettings() const {
  const bool comes_from_file = !input_model_path_.empty();
  const bool comes_from_memory = input_model_data_ != nullptr;

  if (!comes_from_file && !comes_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer");
  }

  if (comes_from_file && comes_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer, ",
                           "but not both.");
  }

  if (comes_from_file && !std::filesystem::exists(input_model_path_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input model path does not exist: ", input_model_path_);
  }

  if (comes_from_memory && input_model_data_size_ == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer for input model data has size 0");
  }

  return Status::OK();
}

Status ModelCompilationOptions::CheckOutputModelSettings() const {
  const EpContextModelGenerationOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;

  const bool explicit_writes_to_file = !ep_context_gen_options.output_model_file_path.empty();
  const bool writes_to_buffer = ep_context_gen_options.output_model_buffer_ptr != nullptr;

  if (!explicit_writes_to_file && !writes_to_buffer) {
    // User did not specify an output file or an output buffer. We default to generating an output file
    // with a name based on the input file name, so do not return an error.
    return Status::OK();
  }

  if (explicit_writes_to_file && writes_to_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Output model to compile must be saved either to a file or to a buffer, but not both.");
  }

  if (writes_to_buffer && ep_context_gen_options.output_model_buffer_size_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: size pointer is null");
  }

  if (writes_to_buffer && ep_context_gen_options.output_model_buffer_allocator == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: allocator is null");
  }

  return Status::OK();
}

Status ModelCompilationOptions::Check() const {
  ORT_ENFORCE(session_options_.value.ep_context_gen_options.enable);
  ORT_ENFORCE(session_options_.value.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableModelCompile, "0") == "0");
  ORT_RETURN_IF_ERROR(CheckInputModelSettings());
  ORT_RETURN_IF_ERROR(CheckOutputModelSettings());
  return Status::OK();
}
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
