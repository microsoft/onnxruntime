// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

enum class GatherBlockQuantizedDataPolicy {
  DeviceCopy,
  DirectHost,
};

constexpr GatherBlockQuantizedDataPolicy SelectGatherBlockQuantizedDataPolicy(
    bool option_enabled, bool pageable_memory_access, bool uses_host_page_tables,
    bool is_fp8, bool is_constant_initializer) {
  return option_enabled && pageable_memory_access && uses_host_page_tables &&
                 is_fp8 && is_constant_initializer
             ? GatherBlockQuantizedDataPolicy::DirectHost
             : GatherBlockQuantizedDataPolicy::DeviceCopy;
}

inline thread_local bool host_pageable_gather_registration_enabled = false;

class ScopedHostPageableGatherRegistration final {
 public:
  explicit ScopedHostPageableGatherRegistration(bool enabled)
      : previous_(host_pageable_gather_registration_enabled) {
    host_pageable_gather_registration_enabled = enabled;
  }

  ~ScopedHostPageableGatherRegistration() {
    host_pageable_gather_registration_enabled = previous_;
  }

  ScopedHostPageableGatherRegistration(const ScopedHostPageableGatherRegistration&) = delete;
  ScopedHostPageableGatherRegistration& operator=(const ScopedHostPageableGatherRegistration&) = delete;

 private:
  bool previous_;
};

}  // namespace onnxruntime::contrib::cuda
