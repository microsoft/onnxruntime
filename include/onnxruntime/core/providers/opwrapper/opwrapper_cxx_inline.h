// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "opwrapper_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

namespace Ort {
namespace OpWrapper {
inline ProviderOptions::ProviderOptions(OrtOpWrapperProviderOptions* options)
    : Base<OrtOpWrapperProviderOptions>{options} {
}

inline ProviderOptions::ProviderOptions() {
  Ort::ThrowOnError(GetApi().CreateProviderOptions(&p_));
}

inline ProviderOptions::ProviderOptions(const std::unordered_map<std::string, std::string>& opts) {
  Ort::ThrowOnError(GetApi().CreateProviderOptions(&p_));
  UpdateOptions(opts);
}

inline size_t ProviderOptions::HasOption(const char* key) const {
  size_t size = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_HasOption(p_, key, &size));
  return size;
}

inline std::string ProviderOptions::GetOption(const char* key, size_t value_size) const {
  std::string value;

  if (value_size == 0) {
    Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, nullptr, &value_size));
  }

  value.resize(value_size);
  Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, &value[0], &value_size));
  value.resize(value_size - 1);  // remove the terminating character '\0'

  return value;
}

inline void ProviderOptions::UpdateOptions(const std::unordered_map<std::string, std::string>& options) {
  const size_t num_options = options.size();
  std::vector<const char*> keys;
  std::vector<const char*> vals;

  keys.reserve(num_options);
  vals.reserve(num_options);

  for (const auto& it : options) {
    keys.push_back(it.first.c_str());
    vals.push_back(it.second.c_str());
  }

  Ort::ThrowOnError(GetApi().ProviderOptions_Update(p_, keys.data(), vals.data(), num_options));
}

inline std::unordered_map<std::string, std::string> ProviderOptions::ToMap() const {
  std::vector<char> all_keys;
  std::vector<char> all_vals;
  size_t num_options = 0;
  size_t keys_size = 0;
  size_t vals_size = 0;

  ThrowOnError(GetApi().ProviderOptions_Serialize(p_, nullptr, &keys_size, nullptr, &vals_size, &num_options));

  all_keys.resize(keys_size);
  all_vals.resize(vals_size);

  ThrowOnError(GetApi().ProviderOptions_Serialize(p_, all_keys.data(), &keys_size, all_vals.data(), &vals_size,
                                                  nullptr));

  std::unordered_map<std::string, std::string> map;
  size_t k_i = 0;
  size_t v_i = 0;

  for (size_t i = 0; i < num_options; ++i) {
    const char* k_cstr = &all_keys.at(k_i);  // If throws out-of-bounds exception, C API has a bug.
    const char* v_cstr = &all_vals.at(v_i);

    std::string key(k_cstr);
    std::string val(v_cstr);
    map.emplace(key, val);

    k_i += key.length() + 1;
    v_i += val.length() + 1;
  }

  return map;
}

inline ProviderOptions ProviderOptions::FromKernelInfo(Unowned<const KernelInfo>& kernel_info,
                                                       const char* op_name) {
  OrtOpWrapperProviderOptions* options = nullptr;
  Ort::ThrowOnError(GetApi().KernelInfo_GetProviderOptions(kernel_info, op_name, &options));
  return ProviderOptions(options);
}

inline Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                                    const std::unordered_map<std::string, ProviderOptions>& provider_options) {
  const size_t num_ops = provider_options.size();
  std::vector<const char*> op_names;
  std::vector<const OrtOpWrapperProviderOptions*> op_options;

  op_names.reserve(num_ops);
  op_options.reserve(num_ops);

  for (const auto& it : provider_options) {
    op_names.push_back(it.first.c_str());
    op_options.push_back(it.second);
  }

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpWrapper(session_options, op_names.data(),
                                                                             op_options.data(), num_ops));

  return session_options;
}

inline Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                                    const char* op_name,
                                                    const ProviderOptions& op_options) {
  constexpr size_t num_ops = 1;
  const OrtOpWrapperProviderOptions* ops_options[num_ops] = {op_options};

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpWrapper(session_options, &op_name,
                                                                             ops_options, num_ops));
  return session_options;
}
}  // namespace OpWrapper
}  // namespace Ort
