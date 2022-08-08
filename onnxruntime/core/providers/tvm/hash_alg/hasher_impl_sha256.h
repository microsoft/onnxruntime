// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_SHA256_H
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_SHA256_H

#include <ippcp.h>

#include "hasher_impl.h"


namespace onnxruntime {
namespace tvm {

class HasherSHA256Impl : public HasherImpl {
 public:
  HasherSHA256Impl() = default;
  virtual ~HasherSHA256Impl() = default;

  std::string hash(const char* src, size_t size) const final;

 private:
  static void digest(const Ipp8u* src, int size, Ipp8u* dst);
  static std::string digest(const char* src, size_t size);
  static std::string hexdigest(const char* src, size_t size);
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_SHA256_H
