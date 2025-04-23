// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"

#include "core/common/logging/logging.h"
#include "core/common/string_utils.h"
#include "core/framework/ort_value.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace {

Status CheckInitializer(const char* name, const OrtValue* val) {
  if (name == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for name");
  }

  if (val == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for OrtValue");
  }

  if (!val->IsTensor()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received OrtValue is not a tensor. Only tensors are supported.");
  }
  if (val->Get<Tensor>().OwnsBuffer()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer containing the initializer must be owned by the user.");
  }
  return Status::OK();
}

}  // namespace

Status SessionOptions::AddInitializer(_In_z_ const char* name, _In_ const OrtValue* val) {
  // input validation
  ORT_RETURN_IF_ERROR(CheckInitializer(name, val));
  // now do the actual work
  bool result = initializers_to_share_map.emplace(name, val).second;

  if (!result) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added: ", name);
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
Status SessionOptions::AddExternalInitializers(gsl::span<const std::string> names, gsl::span<const OrtValue> values) {
  const auto init_num = names.size();
  ORT_ENFORCE(init_num == values.size(), "Expecting same size spans");
  external_initializers.reserve(external_initializers.size() + init_num);
  for (size_t i = 0; i < init_num; ++i) {
    ORT_RETURN_IF_ERROR(CheckInitializer(names[i].c_str(), &values[i]));
    bool result = external_initializers.emplace(names[i], values[i]).second;
    if (!result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added: ", names[i]);
    }
  }
  return Status::OK();
}

Status SessionOptions::AddExternalInitializersFromFilesInMemory(gsl::span<const PathString> file_names,
                                                                gsl::span<std::pair<char*, const size_t>> files_buffers) {
  const auto num_files = file_names.size();
  ORT_ENFORCE(num_files == files_buffers.size(), "Expecting same size spans");
  external_initializer_files_mmap.reserve(external_initializer_files_mmap.size() + num_files);
  static constexpr std::array<std::basic_string_view<ORTCHAR_T>, 4> prefix_list{
      ORT_TSTR(".//"),
      ORT_TSTR("./"),
      ORT_TSTR(".\\\\"),
      ORT_TSTR(".\\")};
  for (size_t i = 0; i < num_files; ++i) {
    // ignore "./" from file name if it has
    auto file_name = file_names[i];
    for (auto prefix : prefix_list) {
      if (file_name.rfind(prefix, 0) == 0) {
        file_name = file_name.substr(prefix.length());
        break;
      }
    }
    bool result = external_initializer_files_mmap.emplace(file_name, files_buffers[i]).second;
    if (!result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An entry for this name has already been added: ",
                             ORT_TSTR_CONVERT_TO_PRINTABLE_STRING(file_name));
    }
  }
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
void SessionOptions::AddCustomOpLibraryHandle(PathString library_name, void* library_handle) {
  if (!this->custom_op_libs) {
    this->custom_op_libs = std::make_shared<LibraryHandles>();
  }

  this->custom_op_libs->Add(std::move(library_name), library_handle);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

EpContextModelGenerationOptions::EpContextModelGenerationOptions(const ConfigOptions& config_options) {
  enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  output_model_file_path = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  output_external_initializers_file_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName, "");
  output_external_initializer_size_threshold = 0;
  embed_ep_context_in_model = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
}

EpContextModelGenerationOptions SessionOptions::GetEpContextGenerationOptions() const {
  if (this->has_explicit_ep_context_gen_options) {
    return this->ep_context_gen_options;
  }

  return EpContextModelGenerationOptions(this->config_options);
}
}  // namespace onnxruntime
