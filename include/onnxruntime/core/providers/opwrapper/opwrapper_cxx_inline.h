// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "opwrapper_cxx_api.h" instead.
//
// These are the inline implementations of the C++ header APIs. They're in this separate file as to not clutter
// the main C++ file with implementation details.

namespace Ort {
namespace OpWrapper {
inline OpWrapperProviderOptions::OpWrapperProviderOptions(OrtOpWrapperProviderOptions* options)
    : Base<OrtOpWrapperProviderOptions>{options} {
}

inline size_t OpWrapperProviderOptions::HasOption(const char* key) const {
  size_t size = 0;
  Ort::ThrowOnError(GetApi().ProviderOptions_HasOption(p_, key, &size));
  return size;
}

inline std::string OpWrapperProviderOptions::GetOption(const char* key, size_t value_size) const {
  std::string value;

  if (value_size == 0) {
    Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, nullptr, &value_size));
  }

  value.resize(value_size);
  Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, &value[0], &value_size));
  value.resize(value_size - 1);  // remove the terminating character '\0'

  return value;
}

inline std::unordered_map<std::string, std::string> OpWrapperProviderOptions::ToMap() const {
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

OpWrapperProviderOptions OpWrapperProviderOptions::FromKernelInfo(Unowned<const KernelInfo> kernel_info,
                                                                  const char* op_name) {
  OrtOpWrapperProviderOptions* options = nullptr;
  Ort::ThrowOnError(GetApi().KernelInfo_GetProviderOptions(kernel_info, op_name, &options));
  return OpWrapperProviderOptions(options);
}
}  // namespace OpWrapper
}  // namespace Ort
