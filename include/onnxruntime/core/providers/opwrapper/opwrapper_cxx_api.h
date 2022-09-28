// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opwrapper_provider_factory.h"
#include "onnxruntime_cxx_api.h"

namespace Ort {
ORT_DEFINE_RELEASE(ProviderOptions);

struct ProviderOptions : Base<OrtOpWrapperProviderOptions> {
  explicit ProviderOptions(std::nullptr_t) {}
  explicit ProviderOptions(OrtOpWrapperProviderOptions* options);

  size_t HasOption(const char* key) const;
  std::string GetOption(const char* key, size_t value_size = 0) const;
  std::unordered_map<std::string, std::string> ToMap() const;

  static ProviderOptions FromKernelInfo(Unowned<const KernelInfo> kernel_info);
};


// TODO: IMPLEMENTATION

inline Ort::ProviderOptions::ProviderOptions(OrtOpWrapperProviderOptions* options) : Base<OrtOpWrapperProviderOptions>{options} {
}

inline size_t Ort::ProviderOptions::HasOption(const char* key) const {
  size_t size = 0;
  //Ort::ThrowOnError(GetApi().ProviderOptions_HasOption(p_, key, &size));
  return size;
}

inline std::string Ort::ProviderOptions::GetOption(const char* key, size_t value_size) const {
  std::string value;
  /*
  if (value_size == 0) {
    Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, nullptr, &value_size));
  }

  value.resize(value_size);
  Ort::ThrowOnError(GetApi().ProviderOptions_GetOption(p_, key, &value[0], &value_size));
  value.resize(value_size - 1);  // remove the terminating character '\0'
  */
  return value;
}

inline std::unordered_map<std::string, std::string> Ort::ProviderOptions::ToMap() const {
  std::vector<char> all_keys;
  std::vector<char> all_vals;
  size_t num_options = 0;
  size_t keys_size = 0;
  size_t vals_size = 0;

  //ThrowOnError(GetApi().ProviderOptions_Serialize(p_, nullptr, &keys_size, nullptr, &vals_size, &num_options));

  all_keys.resize(keys_size);
  all_vals.resize(vals_size);

  //ThrowOnError(GetApi().ProviderOptions_Serialize(p_, all_keys.data(), &keys_size, all_vals.data(), &vals_size,
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


Ort::ProviderOptions ProviderOptions::FromKernelInfo(Unowned<const KernelInfo> kernel_info) {
  OrtProviderOptions* options = nullptr;
  //Ort::ThrowOnError(GetApi().KernelInfo_GetProviderOptions(p_, &options));
  return Ort::ProviderOptions(options);
}

}  // namespace Ort