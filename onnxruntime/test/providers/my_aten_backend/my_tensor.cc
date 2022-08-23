#include "my_tensor.h"
#include "my_allocator.h"

namespace torch_my_kernel_lib {
namespace aten {

MyTensor::MyTensor(void* buffer, const std::vector<int64_t>& sizes, delete_function delete_func) : buffer_(buffer), sizes_(sizes), delete_func_(delete_func) {}

MyTensor::MyTensor(MyTensor&& other) {
  buffer_ = other.buffer_;
  sizes_ = other.sizes_;
  delete_func_ = other.delete_func_;
  other.buffer_ = nullptr;
  other.delete_func_ = nullptr;
  other.sizes_.clear();
}

MyTensor& MyTensor::operator=(MyTensor&& other) {
  if (this != &other) {
    if (buffer_ != nullptr && delete_func_) {
      delete_func_(buffer_);
    }
    buffer_ = other.buffer_;
    sizes_ = other.sizes_;
    delete_func_ = other.delete_func_;
    other.buffer_ = nullptr;
    other.delete_func_ = nullptr;
    other.sizes_.clear();
  }
  return *this;
}

MyTensor::~MyTensor() {
  if (buffer_ != nullptr && delete_func_) {
    delete_func_(buffer_);
  }
}

void MyTensor::resize(const std::vector<int64_t>& sizes, void* buffer) {
  if (buffer_ && delete_func_) {
    delete_func_(buffer_);
  }
  sizes_ = sizes;
  buffer_ = buffer;
}

c10::intrusive_ptr<c10::TensorImpl> MyATenTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<MyATenTensorImpl>(
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

c10::intrusive_ptr<c10::TensorImpl> MyATenTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<MyATenTensorImpl>(
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

void MyATenTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  auto* src_impl = dynamic_cast<MyATenTensorImpl*>(impl.get());
  copy_tensor_metadata(
      src_impl,
      this,
      version_counter(),
      allow_tensor_metadata_change());
}

at::IntArrayRef MyATenTensorImpl::sizes_custom() const {
  const_cast<MyATenTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::sizes_default();
}

int64_t MyATenTensorImpl::dim_custom() const {
  const_cast<MyATenTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::dim_default();
}

int64_t MyATenTensorImpl::numel_custom() const {
  const_cast<MyATenTensorImpl*>(this)->cacheSizeMetadata();
  return c10::TensorImpl::numel_default();
}

bool MyATenTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  return true;
}

void MyATenTensorImpl::cacheSizeMetadata() {
  // TODO: wrap with change generation guard
  const auto& shape = tensor_->sizes();
  // calculate strides
  std::vector<int64_t> strides_vec;
  strides_vec.resize(shape.size());
  int64_t running_size = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    strides_vec[i - 1] = running_size;
    running_size *= shape[i - 1];
  }
  // calcuate num of elements from shape
  numel_ = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    numel_ *= shape[i];
  }

  sizes_and_strides_.set_sizes(c10::IntArrayRef(shape.data(), shape.size()));

  for (std::size_t i = 0; i < strides_vec.size(); i++) {
    sizes_and_strides_.stride_at_unchecked(i) = strides_vec[i];
  }
}

const at::Storage& MyATenTensorImpl::storage() const {
  throw std::runtime_error("ORT Tensors do not have storage");
}

bool MyATenTensorImpl::has_storage() const {
  return false;
}

at::IntArrayRef MyATenTensorImpl::strides_custom() const {
  const_cast<MyATenTensorImpl*>(this)->cacheSizeMetadata();
  return sizes_and_strides_.strides_arrayref();
}

}  // namespace aten
}  // namespace torch_my_kernel_lib
