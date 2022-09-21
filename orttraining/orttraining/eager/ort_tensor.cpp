// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_tensor.h"
#include "ort_util.h"

namespace torch_ort {
namespace eager {

c10::intrusive_ptr<c10::TensorImpl> ORTTensorImpl::shallow_copy_and_detach(
  const c10::VariableVersion& version_counter,
  bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<ORTTensorImpl>(
    tensor_,
    at::TensorOptions()
      .dtype(this->dtype())
      .device(this->device()));

  copy_tensor_metadata(
    this,
    impl.get(),
    version_counter,
    allow_tensor_metadata_change);

  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> ORTTensorImpl::shallow_copy_and_detach(
  c10::VariableVersion&& version_counter,
  bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<ORTTensorImpl>(
    tensor_,
    at::TensorOptions()
      .dtype(this->dtype())
      .device(this->device()));

  copy_tensor_metadata(
    this,
    impl.get(),
    std::move(version_counter),
    allow_tensor_metadata_change);

  return impl;
}

void ORTTensorImpl::shallow_copy_from(
  const c10::intrusive_ptr<TensorImpl>& impl) {
  auto* src_impl = dynamic_cast<ORTTensorImpl*>(impl.get());
  copy_tensor_metadata(
    src_impl,
    this,
    version_counter(),
    allow_tensor_metadata_change());
}

at::IntArrayRef ORTTensorImpl::sizes_custom() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::sizes_default();
}

int64_t ORTTensorImpl::dim_custom() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::dim_default();
}

int64_t ORTTensorImpl::numel_custom() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::numel_default();
}

bool ORTTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  auto& tensor = tensor_.Get<onnxruntime::Tensor>();
  return tensor.IsContiguous();
}

void ORTTensorImpl::cacheSizeMetadata() {
  // TODO: wrap with change generation guard
  auto& tensor = tensor_.Get<onnxruntime::Tensor>();
  const auto& shape = tensor.Shape();
  const auto dims = shape.GetDims();
  auto strides = tensor.Strides();

  numel_ = shape.Size();

  sizes_and_strides_.set_sizes(c10::IntArrayRef(dims.data(), dims.size()));

  for (std::size_t i = 0; i < strides.size(); i++) {
    sizes_and_strides_.stride_at_unchecked(i) = strides[i];
  }
}

const at::Storage& ORTTensorImpl::storage() const {
  throw std::runtime_error("ORT Tensors do not have storage");
}

bool ORTTensorImpl::has_storage() const {
  return false;
}

at::IntArrayRef ORTTensorImpl::strides_custom() const {
  const_cast<ORTTensorImpl*>(this)->cacheSizeMetadata();
  return sizes_and_strides_.strides_arrayref();
}

} // namespace eager
} // namespace torch_ort
