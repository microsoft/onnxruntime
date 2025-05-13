// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/model_compilation_options.h"

#include <memory>
#include <string>
#include <utility>

#include "core/framework/allocator.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/environment.h"

namespace onnxruntime {
ModelCompilationOptions::ModelCompilationOptions(const onnxruntime::Environment& env, const OrtSessionOptions& session_options)
    : env_(env), session_options_(session_options) {
  session_options_.value.has_explicit_ep_context_gen_options = true;
  session_options_.value.ep_context_gen_options = session_options.value.GetEpContextGenerationOptions();
  session_options_.value.ep_context_gen_options.enable = true;
  session_options_.value.ep_context_gen_options.overwrite_existing_output_file = true;
  // defaulting to false to support wider usage. will log WARNING if compiling model with no context nodes.
  // TODO: Add ability for user to explicitly set this.
  session_options_.value.ep_context_gen_options.error_if_no_compiled_nodes = false;

  // Shouldn't fail because the key/value strings are below the maximum string length limits in ConfigOptions.
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1").IsOK());
  ORT_ENFORCE(session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "0").IsOK());
}

void ModelCompilationOptions::SetInputModelPath(const std::string& input_model_path) {
  input_model_variant_ = input_model_path;
}

void ModelCompilationOptions::SetInputModelFromBuffer(const void* input_model_data, size_t input_model_data_size) {
  input_model_variant_ = gsl::span<const std::byte>(reinterpret_cast<const std::byte*>(input_model_data),
                                                    input_model_data_size);
}

void ModelCompilationOptions::SetInputOrtModel(const OrtModel& ort_model) {
  input_model_variant_ = &ort_model;
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

Status ModelCompilationOptions::SetOutputModelBuffer(onnxruntime::AllocatorPtr allocator,
                                                     void** output_model_buffer_ptr,
                                                     size_t* output_model_buffer_size_ptr) {
  ORT_RETURN_IF_ERROR(ResetOutputModelSettings());

  session_options_.value.ep_context_gen_options.output_model_buffer_ptr = output_model_buffer_ptr;
  session_options_.value.ep_context_gen_options.output_model_buffer_size_ptr = output_model_buffer_size_ptr;
  session_options_.value.ep_context_gen_options.output_model_buffer_allocator = std::move(allocator);
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

const std::string* ModelCompilationOptions::TryGetInputModelPath() const {
  return std::get_if<std::string>(&input_model_variant_);
}

const gsl::span<const std::byte>* ModelCompilationOptions::TryGetInputModelBuffer() const {
  return std::get_if<gsl::span<const std::byte>>(&input_model_variant_);
}

const OrtModel* ModelCompilationOptions::TryGetInputOrtModel() const {
  const gsl::not_null<const OrtModel*>* ort_model_ptr_ptr = std::get_if<gsl::not_null<const OrtModel*>>(
      &input_model_variant_);
  return (ort_model_ptr_ptr == nullptr) ? nullptr : ort_model_ptr_ptr->get();
}

Status ModelCompilationOptions::ResetOutputModelSettings() {
  EpContextModelGenerationOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;
  ep_context_gen_options.output_model_file_path.clear();
  ep_context_gen_options.output_model_buffer_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_size_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_allocator = nullptr;
  return session_options_.value.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");
}

Status ModelCompilationOptions::Check() const {
  ORT_ENFORCE(session_options_.value.ep_context_gen_options.enable);
  ORT_ENFORCE(session_options_.value.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableModelCompile, "0") == "0");

  // Check input model settings
  if (std::holds_alternative<std::monostate>(input_model_variant_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file, a memory buffer, ",
                           "or an OrtModel instance");
  }

  const std::string* input_model_path_ptr = TryGetInputModelPath();
  const gsl::span<const std::byte>* input_model_buffer_ptr = TryGetInputModelBuffer();
  const OrtModel* ort_model = TryGetInputOrtModel();

  if (input_model_path_ptr != nullptr && !std::filesystem::exists(*input_model_path_ptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input model path does not exist: ", *input_model_path_ptr);
  }

  if (input_model_buffer_ptr != nullptr && input_model_buffer_ptr->size() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer for input model data has size 0");
  }

  if (ort_model != nullptr && ort_model->graph->nodes.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input OrtModel instance has no nodes");
  }

  // Check output model settings
  const EpContextModelGenerationOptions& ep_context_gen_options = session_options_.value.ep_context_gen_options;

  const bool explicit_output_to_file = !ep_context_gen_options.output_model_file_path.empty();
  const bool output_to_buffer = ep_context_gen_options.output_model_buffer_ptr != nullptr;

  if (!explicit_output_to_file && !output_to_buffer && input_model_path_ptr != nullptr) {
    // User did not specify an output file or an output buffer. We default to generating an output file
    // with a name based on the input file name, so do not return an error.
    return Status::OK();
  }

  if (!explicit_output_to_file && !output_to_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unable to generate an output model path: require an input model path if the location "
                           "of the output model (e.g., file or buffer) is not specified.");
  }

  if (explicit_output_to_file && output_to_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Output model to compile must be saved either to a file or to a buffer, but not both.");
  }

  if (output_to_buffer && ep_context_gen_options.output_model_buffer_size_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: size pointer is null");
  }

  if (output_to_buffer && ep_context_gen_options.output_model_buffer_allocator == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid buffer configuration for output model: allocator is null");
  }

  return Status::OK();
}
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
