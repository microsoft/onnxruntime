// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/common/safeint.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/utils.h"
#include "core/framework/storage.h"

namespace onnxruntime {

static auto MakeBuffer(std::size_t sizeInBytes, AllocatorPtr allocatorPtr, MemoryLocation location=MemoryLocation::Uniform, std::size_t offset=0)
{
    auto deleter = [&allocatorPtr](void *ptr){ allocatorPtr->Free(ptr); };
    return Buffer(sizeInBytes, location, offset, allocatorPtr->Alloc(sizeInBytes), deleter);
}

static auto MakeBuffer(std::size_t sizeInBytes, void* ptr, DeleterFnPtr deleter=nullptr, MemoryLocation location=MemoryLocation::Uniform, std::size_t offset=0)
{
    return Buffer(sizeInBytes, location, offset, ptr, deleter);
}

#ifdef ENABLE_STRIDED_TENSORS
namespace {
int64_t GetSizeFromStrides(const TensorShape& shape, gsl::span<const int64_t> strides) {
  SafeInt<int64_t> size = 1;
  for (size_t dim = 0; dim < shape.NumDimensions(); ++dim) {
    if (shape[dim] == 0) {
      size = 0;
      break;
    }
    size += strides[dim] * (shape[dim] - 1);
  }
  return size;
}
}  // namespace
#endif

size_t Tensor::CalculateTensorStorageSize(MLDataType p_type,
                                          const TensorShape& shape,
                                          gsl::span<const int64_t> strides) {
#ifdef ENABLE_STRIDED_TENSORS
  int64_t shape_size = 1;
  if (shape.NumDimensions() > 0 && !strides.empty()) {
    ORT_ENFORCE(shape.NumDimensions() == strides.size(), "Length of strides doesn't match with tensor dimension size.");
    shape_size = GetSizeFromStrides(shape, strides);
  } else {
    shape_size = shape.Size();
  }
#else
  ORT_ENFORCE(strides.empty(), "Strided tensor is supported for training only for now.");
  int64_t shape_size = shape.Size();
#endif
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");

  if (shape_size > 0) {
    SafeInt<size_t> len = 0;
    if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_size), p_type->Size(), &len))
      ORT_THROW("tensor failed memory size calculation");

    return len;
  }
  return 0;
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
               ptrdiff_t offset, gsl::span<const int64_t> strides)
    : alloc_info_(alloc) {
  ORT_ENFORCE(p_type != nullptr);
  const size_t len = Tensor::CalculateTensorStorageSize(p_type, shape, strides);
  auto buffer = MakeBuffer(len, p_data, nullptr /*deleter*/, alloc_info_.location, offset);
  Init(p_type, std::make_shared<Storage>(std::move(buffer)), shape, strides);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator,
               gsl::span<const int64_t> strides)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  size_t len = Tensor::CalculateTensorStorageSize(p_type, shape, strides);
  auto buffer = MakeBuffer(len, allocator, alloc_info_.location, 0L);
  Init(p_type, std::make_shared<Storage>(std::move(buffer)), shape, strides);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
               ptrdiff_t offset, gsl::span<const int64_t> strides)
    : alloc_info_(deleter->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  size_t len = Tensor::CalculateTensorStorageSize(p_type, shape, strides);
  auto buffer = MakeBuffer(len, p_data, [deleter](void *ptr){ deleter->Free(ptr); }, alloc_info_.location, offset);
  Init(p_type, std::make_shared<Storage>(std::move(buffer)), shape, strides);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::vector<std::shared_ptr<IAllocator>> allocators,
               gsl::span<const int64_t> strides)
    : Tensor(!shape.shardInfo().has_value() && allocators.size() == 1? Tensor(p_type, shape, allocators[0], strides) : Tensor()) {
  alloc_info_ = allocators[0]->Info();
  ORT_ENFORCE(p_type != nullptr);
  ORT_ENFORCE(shape.shardInfo().has_value());
  ORT_ENFORCE(allocators.size() > 1);
  auto shardDims = shape.shardInfo()->shardDims_;
  assert(ShardUtils::NumShards(shardDims) == allocators.size());
  std::vector<Buffer> buffers;
  for (auto allocator : allocators)
  {
    TensorShape shard = ShardUtils::GetShardSize(shape, shardDims);
    const size_t len = Tensor::CalculateTensorStorageSize(p_type, shard, strides);
    buffers.emplace_back(MakeBuffer(len, allocator, allocator->Info().location));
  }
  Init(p_type, std::make_shared<Storage>(std::move(buffers), std::make_optional(shardDims)), shape, strides);
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::vector<std::shared_ptr<IAllocator>> allocators,
                          OrtValue& ort_value, gsl::span<const int64_t> strides) {
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, std::move(allocators), strides);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator,
                          OrtValue& ort_value, gsl::span<const int64_t> strides) {
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, std::move(allocator), strides);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location,
                          OrtValue& ort_value, ptrdiff_t offset, gsl::span<const int64_t> strides) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(p_type, shape, p_data, location, offset, strides);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType p_type, const TensorShape& shape,
                          void* p_data, std::shared_ptr<IAllocator> allocator,
                          OrtValue& ort_value, ptrdiff_t offset,
                          gsl::span<const int64_t> strides) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(p_type, shape, p_data, std::move(allocator), offset, strides);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(Tensor&& tensor, OrtValue& ort_value) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(std::move(tensor));
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

size_t Tensor::SizeInBytes() const {
#ifdef ENABLE_STRIDED_TENSORS
  int64_t size = IsContiguous() ? shape_.Size() : GetSizeFromStrides(shape_, strides_);
#else
  int64_t size = shape_.Size();
#endif
  size_t ret;
  if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(size), dtype_->Size(), &ret)) {
    ORT_THROW("tensor size overflow");
  }
  return ret;
}

void Tensor::Init(MLDataType p_type, std::shared_ptr<Storage> storage, const TensorShape& shape,
            gsl::span<const int64_t> strides) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");
  dtype_ = p_type->AsPrimitiveDataType();
  ORT_ENFORCE(dtype_ != nullptr,
              "Tensor is expected to contain one of the primitive data types. Got: ", DataTypeImpl::ToString(p_type));
  shape_ = shape;
  storage_ = storage;

  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (IsDataTypeString())
  {
    storage_->Apply([=, elemSize = p_type->Size()](Buffer& buffer){
      if (buffer.OwnBuffer()) {
        utils::ConstructStrings(buffer.Data(), buffer.Size() / elemSize);
      }
    });
  }
#ifdef ENABLE_STRIDED_TENSORS
  if (shape.NumDimensions() > 0 && !strides.empty()) {
    ORT_ENFORCE(shape.NumDimensions() == strides.size(), "Length of strides doesn't match with tensor dimension size.");
    strides_.assign(strides.begin(), strides.end());
    is_contiguous_ = CheckIsContiguous();
  }
#else
  ORT_UNUSED_PARAMETER(strides);
#endif
}

Tensor::Tensor(Tensor&& other) noexcept
    : storage_(other.storage_),
      shape_(other.shape_),
      dtype_(other.dtype_),
      alloc_info_(other.alloc_info_) {
  other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
  other.storage_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    ReleaseBuffer();

    dtype_ = other.dtype_;
    shape_ = other.shape_;
    storage_ = other.storage_;
    alloc_info_ = other.alloc_info_;


    other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.storage_ = nullptr;
  }
  return *this;
}

Tensor::~Tensor() {
  ReleaseBuffer();
}

void Tensor::ReleaseBuffer() {
  if (IsDataTypeString()) {
    storage_->Apply([=, elemSize = this->DataType()->Size()](Buffer& buffer){
      if (buffer.OwnBuffer()) {
            utils::DestroyStrings(buffer.Data(), buffer.Size() / elemSize);
          }
    });
  }
  storage_.reset();
}

#ifdef ENABLE_STRIDED_TENSORS
bool Tensor::CheckIsContiguous() const {
  if (strides_.empty()) {
    return true;
  }

  int64_t running_size = 1;
  for (size_t i = shape_.NumDimensions(); i > 0; --i) {
    size_t j = i - 1;
    if (shape_[j] == 0) {
      return true;
    }

    if (shape_[j] != 1 && strides_[j] != running_size) {
      return false;
    }

    running_size *= shape_[j];
  }

  return true;
}

gsl::span<const int64_t> Tensor::Strides() const {
  if (shape_.NumDimensions() == 0) {
    return {};
  }

  if (strides_.empty()) {
    strides_.resize(shape_.NumDimensions());
    int64_t running_size = 1;
    for (size_t i = shape_.NumDimensions(); i > 0; --i) {
      strides_[i - 1] = running_size;
      running_size *= shape_[i - 1];
    }
  }

  return gsl::make_span(strides_);
}

void Tensor::SetShapeAndStrides(const TensorShape& new_shape, gsl::span<const int64_t> new_strides) {
  ORT_ENFORCE(new_shape.NumDimensions() == new_strides.size(),
              "Length of strides doesn't match with tensor dimension size.");
  shape_ = new_shape;
  strides_ = ToShapeVector(new_strides);
  is_contiguous_ = CheckIsContiguous();
}
#endif

}  // namespace onnxruntime
