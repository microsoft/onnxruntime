// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hasher_impl.h"  // NOLINT(build/include_subdir)
#include "sha256.h"  // NOLINT(build/include_subdir)

namespace onnxruntime {
namespace tvm {

std::string HasherSHA256Impl::hash(const char* src, size_t size) const {
  return SHA256(src, size);
}

}   // namespace tvm
}   // namespace onnxruntime
