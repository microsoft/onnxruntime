// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"

#include "hasher.h"       // NOLINT(build/include_subdir)
#include "hasher_impl.h"  // NOLINT(build/include_subdir)

namespace onnxruntime {
namespace tvm {

Hasher::Hasher(const std::string& hash_type) {
  hasher_ = getHasherImpl(hash_type);
}

std::string Hasher::hash(const char* src, size_t size) const {
  return hasher_->hash(src, size);
}

std::shared_ptr<HasherImpl> Hasher::getHasherImpl(const std::string& hash_type) {
  if (hash_type == "sha256") {
    return std::make_shared<HasherSHA256Impl>();
  } else {
    ORT_NOT_IMPLEMENTED("Hasher was not implemented for hash type: ", hash_type);
  }
  return nullptr;
}

}  // namespace tvm
}  // namespace onnxruntime
