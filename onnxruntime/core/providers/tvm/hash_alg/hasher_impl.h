// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_H_
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_H_

#include <string>


namespace onnxruntime {
namespace tvm {

class HasherImpl {
 public:
  HasherImpl() = default;
  virtual ~HasherImpl() = default;

  virtual std::string hash(const char* src, size_t size) const = 0;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_IMPL_H_
