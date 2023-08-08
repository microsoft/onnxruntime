// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_H_
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_H_

#include <memory>
#include <string>

namespace onnxruntime {
namespace tvm {
class HasherImpl;

class Hasher {
 public:
  Hasher() = delete;
  explicit Hasher(const std::string& hash_type);
  virtual ~Hasher() = default;

  std::string hash(const char* src, size_t size) const;

 private:
  std::shared_ptr<HasherImpl> getHasherImpl(const std::string& hash_type);

 private:
  std::shared_ptr<HasherImpl> hasher_;
};

}  // namespace tvm
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_HASHER_H_
