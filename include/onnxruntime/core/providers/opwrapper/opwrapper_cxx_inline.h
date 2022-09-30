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

inline bool ProviderOptions::HasOption(const char* key) const {
  int exists = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_HasOption(p_, key, &exists));
  return exists != 0;
}

inline std::string_view ProviderOptions::GetOption(const char* key) const {
  const char* out = nullptr;
  size_t len = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, &out, &len));
  return std::string_view(out, len);
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

inline std::unordered_map<std::string_view, std::string_view> ProviderOptions::ToMap(OrtAllocator* allocator) const {
  size_t num_options = 0;
  const char** keys = nullptr;
  const char** vals = nullptr;
  size_t* key_lens = nullptr;
  size_t* val_lens = nullptr;

  ThrowOnError(GetApi().ProviderOptions_Serialize(p_, allocator, &keys, &key_lens, &vals, &val_lens, &num_options));

  std::unordered_map<std::string_view, std::string_view> map;

  for (size_t i = 0; i < num_options; ++i) {
    std::string_view key(keys[i], key_lens[i]);
    std::string_view val(vals[i], val_lens[i]);
    map.emplace(key, val);
  }

  allocator->Free(allocator, keys);
  allocator->Free(allocator, vals);
  allocator->Free(allocator, key_lens);
  allocator->Free(allocator, val_lens);

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

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(session_options, op_names.data(),
                                                                   op_options.data(), num_ops));

  return session_options;
}

inline Ort::SessionOptions& AppendExecutionProvider(Ort::SessionOptions& session_options,
                                                    const char* op_name,
                                                    const ProviderOptions& op_options) {
  constexpr size_t num_ops = 1;
  const OrtOpWrapperProviderOptions* ops_options[num_ops] = {op_options};

  Ort::ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(session_options, &op_name,
                                                                   ops_options, num_ops));
  return session_options;
}
}  // namespace OpWrapper
}  // namespace Ort
