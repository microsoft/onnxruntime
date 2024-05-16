#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  TensorWrapper(onnxruntime::Tensor* tensor) : tensor_(tensor) {}

  // non copyable
  TensorWrapper(const TensorWrapper&) = delete;
  TensorWrapper& operator=(const TensorWrapper&) = delete;

  onnxruntime::Tensor* tensor_ = nullptr;

  int64_t NumberOfElements() const {
    const TensorShape& shape = tensor_->Shape();
    return shape.Size();
  }

  int64_t Size(int i) const {
    const TensorShape& shape = tensor_->Shape();
    if (i < 0) {
      assert(shape.NumDimensions() + i >= 0);
      return shape[shape.NumDimensions() + i];
    }

    return shape[i];
  }

  int64_t Stride(int i) const {
    const TensorShape& shape = tensor_->Shape();
    assert(i >= 0 && static_cast<size_t>(i) < shape.NumDimensions());
    return shape.SizeFromDimension(i + 1);
  }

  size_t Dim() const {
    return tensor_->Shape().NumDimensions();
  }

  bool HasValue() const {
    return tensor_ != nullptr;
  }

  // Checks if the Tensor contains data type T
  template <class T>
  bool IsDataType() const {
    return utils::IsPrimitiveDataType<T>(tensor_->DataType());
  }

 template <typename T>
  const T* Data() const {
    return tensor_->Data<T>();
  }

template <typename T>
  T* MutableData() {
    return tensor_->MutableData<T>();
  }

  // static Empty(std::initializer_list<int64_t> dims){
  //     Tensor(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator);
  // }
};
};  // namespace onnxruntime
