// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <c10/core/TensorImpl.h>
#include <core/framework/ort_value.h>

namespace torch_ort {
namespace eager {

class ORTTensorImpl final : public c10::TensorImpl {
 public:
  explicit ORTTensorImpl(OrtValue tensor, const at::TensorOptions& options)
    : c10::TensorImpl(
        c10::DispatchKeySet {
          c10::DispatchKey::ORT,
          c10::DispatchKey::AutogradORT
        },
        options.dtype(),
        options.device()) {
    set_tensor(tensor);
  }

  OrtValue& tensor() {
    return tensor_;
  }

  void set_tensor(OrtValue tensor) {
    tensor_ = std::move(tensor);
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes() const override;

  int64_t dim() const override;

  int64_t numel() const override;

  bool is_contiguous(at::MemoryFormat memory_format) const override;

  int64_t size(int64_t d) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

  at::IntArrayRef strides() const override;  

 private:
  void cacheSizeMetadata();
  OrtValue tensor_;
};

} // namespace eager
} // namespace torch_ort