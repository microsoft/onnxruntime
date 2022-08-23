// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <c10/core/TensorImpl.h>

namespace torch_my_kernel_lib {
namespace aten {

using delete_function = std::function<void(void*)>;

class MyTensor {
 public:
  MyTensor(void* buffer, const std::vector<int64_t>& sizes, delete_function delete_func);
  MyTensor(const MyTensor&) = delete;
  MyTensor(MyTensor&& other);
  MyTensor& operator=(const MyTensor&) = delete;
  MyTensor& operator=(MyTensor&& other);
  ~MyTensor();
  void* buffer() const { return buffer_; }
  const std::vector<int64_t>& sizes() const { return sizes_; }

  void resize(const std::vector<int64_t>& sizes, void* buffer);

 private:
  void* buffer_;
  std::vector<int64_t> sizes_;
  delete_function delete_func_;
};

class MyATenTensorImpl final : public c10::TensorImpl {
 public:
  explicit MyATenTensorImpl(std::shared_ptr<MyTensor> tensor, const at::TensorOptions& options)
      : c10::TensorImpl(
            c10::DispatchKeySet{c10::DispatchKey::ORT},
            options.dtype(),
            options.device()) {
    set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
    set_tensor(tensor);
  }

  const MyTensor& tensor() const {
    return *tensor_;
  }

  MyTensor* mutable_tensor() {
    return tensor_.get();
  }

  void set_tensor(std::shared_ptr<MyTensor> tensor) {
    tensor_ = std::move(tensor);
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

  at::IntArrayRef strides_custom() const override;

 private:
  void cacheSizeMetadata();
  std::shared_ptr<MyTensor> tensor_;
};

}  // namespace aten
}  // namespace torch_my_kernel_lib
