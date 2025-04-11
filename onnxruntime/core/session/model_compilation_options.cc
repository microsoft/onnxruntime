// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/model_compilation_options.h"

namespace onnxruntime {
void ModelCompilationOptions::ResetInputModelSettings() {
  input_model_path = "";
  input_model_data = nullptr;
  input_model_data_size = 0;
}

Status ModelCompilationOptions::ResetOutputModelSettings() {
  EpContextModelGenerationOptions& ep_context_gen_options = session_options->value.ep_context_gen_options;
  ep_context_gen_options.output_model_file_path = "";
  ep_context_gen_options.output_model_buffer_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_size_ptr = nullptr;
  ep_context_gen_options.output_model_buffer_allocator = nullptr;
  return session_options->value.config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");
}

Status ModelCompilationOptions::CheckInputModelSettings() const {
  const bool comes_from_file = !input_model_path.empty();
  const bool comes_from_memory = input_model_data != nullptr;

  if (!comes_from_file && !comes_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer");
  }

  if (comes_from_file && comes_from_memory) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input model to compile must be loaded from either a file or a memory buffer, ",
                           "but not both.");
  }

  if (comes_from_file && !std::filesystem::exists(input_model_path)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input model path does not exist: ", input_model_path);
  }

  if (comes_from_memory && input_model_data_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer for input model data has size 0");
  }

  return Status::OK();
}

Status ModelCompilationOptions::CheckOutputModelSettings() const {
  const EpContextModelGenerationOptions& ep_context_gen_options = session_options->value.ep_context_gen_options;

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
  ORT_ENFORCE(session_options != nullptr);
  ORT_ENFORCE(session_options->value.ep_context_gen_options.enable);
  ORT_RETURN_IF_ERROR(CheckInputModelSettings());
  ORT_RETURN_IF_ERROR(CheckOutputModelSettings());
  return Status::OK();
}
}  // namespace onnxruntime
