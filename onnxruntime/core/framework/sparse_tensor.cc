// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/sparse_tensor.h"

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, SparseFormat flags) {
  return os << std::hex << static_cast<uint32_t>(flags);
}

namespace {
// Round up size to a multiple of int64
constexpr size_t kIndexAlignment = alignof(int64_t);
constexpr inline int64_t Roundup(int64_t size) {
  return ((SafeInt<int64_t>(size) + kIndexAlignment - 1) / kIndexAlignment) * kIndexAlignment;
}

/// <summary>
/// Calculate required buffer size. We will place indices
/// after data and make sure indices start at int64_t aligned place
/// </summary>
/// <returns></returns>
constexpr inline int64_t CalculateRequiredBufferSize(int64_t data_size, int64_t indices_size) {
  return SafeInt<int64_t>(Roundup(data_size)) + indices_size;
}

template <typename... T>
inline std::vector<std::reference_wrapper<Tensor>> MakeListNonConst(T&... t) {
  return std::vector{std::ref(t)...};
}

template <typename... T>
inline std::vector<std::reference_wrapper<const Tensor>> MakeListConst(const T&... t) {
  return std::vector{std::cref(t)...};
}

void CopyStrings(const Tensor& src_t, Tensor& dst_t) {
  auto src_span = src_t.DataAsSpan<std::string>();
  std::string* dst = dst_t.MutableData<std::string>();
  std::copy(src_span.begin(), src_span.end(), dst);
}

Status CopyData(const IDataTransfer* data_transfer,
                const std::vector<std::reference_wrapper<const Tensor>>& src,
                const std::vector<std::reference_wrapper<Tensor>>& dst) {
  ORT_RETURN_IF_NOT(src.size() == dst.size(), "Must have the same size. Got src_size: ",
                    src.size(), " dst_size: ", dst.size());
  for (size_t i = 0, src_size = src.size(); i < src_size; ++i) {
    const Tensor& src_t = src[i];
    Tensor& dst_t = dst[i];
    if (src_t.IsDataTypeString()) {
      CopyStrings(src_t, dst_t);
    } else {
      if (data_transfer != nullptr) {
        ORT_RETURN_IF_ERROR(data_transfer->CopyTensor(src_t, dst_t));
      } else {
        memcpy(dst_t.MutableDataRaw(), src_t.DataRaw(), src_t.SizeInBytes());
      }
    }
  }
  return Status::OK();
}

Status CopyStringsAndIndices(size_t string_count, const char* const strings[], Tensor& values,
                             const std::vector<std::reference_wrapper<const Tensor>>& src_ind,
                             const std::vector<std::reference_wrapper<Tensor>>& dst_ind) {
  auto* str_dest = values.MutableData<std::string>();
  for (size_t i = 0; i < string_count; ++i) {
    str_dest[i] = strings[i];
  }

  return CopyData(nullptr, src_ind, dst_ind);
}
}  // namespace

const void* SparseTensor::IndicesStart(int64_t values_bytes) const {
  if (p_data_ != nullptr) {
    return reinterpret_cast<const uint8_t*>(p_data_) + Roundup(values_bytes);
  }
  return nullptr;
}

void* SparseTensor::IndicesStart(int64_t values_bytes) {
  if (p_data_ != nullptr) {
    return reinterpret_cast<uint8_t*>(p_data_) + Roundup(values_bytes);
  }
  return nullptr;
}

int64_t SparseTensor::RequiredAllocationSize() const noexcept {
  if (p_data_ != nullptr) {
    // Can be zero for zero matrices
    assert(buffer_size_ >= 0);
    return buffer_size_;
  } else {
    auto data_size = values_.SizeInBytes();
    int64_t index_size = 0;
    for (const auto& t : format_data_) {
      index_size += t.SizeInBytes();
    }
    return CalculateRequiredBufferSize(data_size, index_size);
  }
}

SparseTensor::SparseTensor(MLDataType elt_type, const TensorShape& dense_shape,
                           const TensorShape& values_shape, void* values_data,
                           const OrtMemoryInfo& location)
    : SparseTensor() {
  dense_shape_ = dense_shape;
  ml_data_type_ = elt_type->AsPrimitiveDataType();
  location_ = location;
  values_ = Tensor(elt_type, values_shape, values_data, location_);
}

SparseTensor::SparseTensor(MLDataType elt_type, const TensorShape& dense_shape,
                           std::shared_ptr<IAllocator> allocator)
    : SparseTensor() {
  dense_shape_ = dense_shape;
  ml_data_type_ = elt_type->AsPrimitiveDataType();
  allocator_ = std::move(allocator);
  location_ = allocator_->Info();
}

SparseTensor::SparseTensor() noexcept
    : format_(SparseFormat::kUndefined),
      dense_shape_(),
      ml_data_type_(nullptr),
      allocator_(),
      location_(),
      p_data_(nullptr),
      buffer_size_(0),
      values_(),
      format_data_() {
}

SparseTensor::SparseTensor(SparseTensor&& o) noexcept : SparseTensor() {
  *this = std::move(o);
}

SparseTensor& SparseTensor::operator=(SparseTensor&& o) noexcept {
  ReleaseBuffer();
  format_ = o.format_;
  dense_shape_ = std::move(o.dense_shape_);
  ml_data_type_ = o.ml_data_type_;
  allocator_ = std::move(o.allocator_);
  location_ = std::move(o.location_);
  std::swap(p_data_, o.p_data_);
  std::swap(buffer_size_, o.buffer_size_);
  values_ = std::move(o.values_);
  format_data_ = std::move(o.format_data_);
  return *this;
}

SparseTensor::~SparseTensor() {
  ReleaseBuffer();
}

void SparseTensor::InitOrtValue(MLDataType elt_type,
                                const TensorShape& dense_shape,
                                const TensorShape& values_shape,
                                void* values_data,
                                const OrtMemoryInfo& location,
                                OrtValue& ort_value) {
  auto sparse_tensor = std::make_unique<SparseTensor>(elt_type, dense_shape, values_shape, values_data, location);
  auto ml_tensor = DataTypeImpl::GetType<SparseTensor>();
  ort_value.Init(sparse_tensor.release(),
                 ml_tensor,
                 ml_tensor->GetDeleteFunc());
}

void SparseTensor::InitOrtValue(MLDataType elt_type,
                                const TensorShape& dense_shape,
                                std::shared_ptr<IAllocator> allocator,
                                OrtValue& ort_value) {
  auto sparse_tensor = std::make_unique<SparseTensor>(elt_type, dense_shape, std::move(allocator));
  auto ml_tensor = DataTypeImpl::GetType<SparseTensor>();
  ort_value.Init(sparse_tensor.release(),
                 ml_tensor,
                 ml_tensor->GetDeleteFunc());
}

const SparseTensor& SparseTensor::GetSparseTensorFromOrtValue(const OrtValue& v) {
  if (!v.IsAllocated()) {
    ORT_THROW("the ort_value must contain a constructed sparse tensor");
  }
  const auto& sparse_tensor = v.Get<onnxruntime::SparseTensor>();
  if (sparse_tensor.Format() == onnxruntime::SparseFormat::kUndefined) {
    ORT_THROW("Sparse Tensor does not contain sparse data");
  }
  return sparse_tensor;
}

SparseTensor& SparseTensor::GetSparseTensorFromOrtValue(OrtValue& v) {
  if (!v.IsAllocated()) {
    ORT_THROW("the ort_value must contain a constructed sparse tensor");
  }
  auto& sparse_tensor = *v.GetMutable<SparseTensor>();
  if (sparse_tensor.Format() != SparseFormat::kUndefined) {
    ORT_THROW("this tensor already has populated sparse_indices");
  }
  return sparse_tensor;
}

Status SparseTensor::AllocateBuffer(int64_t buffer_size, size_t num_values) {
  if (buffer_size > 0) {
    SafeInt<size_t> buffer_size_t(buffer_size);
    const auto values_bytes = SafeInt<size_t>(num_values) * ml_data_type_->Size();
    ORT_RETURN_IF_NOT(buffer_size_t > values_bytes,
                      "Values size ", static_cast<size_t>(values_bytes), " must be less than total buffer size: ", buffer_size);
    auto data_ptr = IAllocator::MakeUniquePtr<void>(allocator_, buffer_size_t);
    if (IsDataTypeString()) {
      // We own the buffer, so we must properly construct strings. Neither of the Tensors
      // we construct on top of the buffer own it. We are constructing empty strings, hopefully
      // nothrow and no buffer allocation
      utils::ConstructStrings(data_ptr.get(), narrow<int64_t>(num_values));
    }
    p_data_ = data_ptr.release();
  }
  buffer_size_ = buffer_size;
  return Status::OK();
}

void SparseTensor::ReleaseBuffer() {
  if (allocator_ && p_data_ != nullptr) {
    if (IsDataTypeString()) {
      utils::DestroyStrings(p_data_, values_.Shape().Size());
    }
    allocator_->Free(p_data_);
  }
  p_data_ = nullptr;
  buffer_size_ = 0;
}

SparseTensor::CooView SparseTensor::AsCoo() const {
  ORT_ENFORCE(Format() == SparseFormat::kCoo, "Must contain Coo format. Got: ", Format());
  ORT_ENFORCE(format_data_.size() == 1U, "Expecting to contain one index, got: ", format_data_.size());
  return CooView(format_data_[0]);
}

std::vector<int64_t> SparseTensor::GetCooIndexDims(size_t values_count, size_t index_size) const {
  std::vector<int64_t> index_dims{narrow<int64_t>(values_count)};
  if (values_count * 2 == index_size) {
    // 2-D COO index
    index_dims.push_back(2);
  } else {
    ORT_ENFORCE(values_count == index_size,
                "Index size: ", index_size, " must be equal to or twice the values size: ", values_count);
  }
  return index_dims;
}

void SparseTensor::InitCooIndex(const TensorShape& index_shape, int64_t* index_data) {
  format_data_.resize(1);
  format_data_[0] = Tensor(DataTypeImpl::GetType<int64_t>(), index_shape,
                           index_data, Location());
  format_ = SparseFormat::kCoo;
}

Status SparseTensor::UseCooIndices(gsl::span<int64_t> indices) {
  ORT_RETURN_IF_NOT(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Not expecting an allocator set");
  TensorShape index_shape(GetCooIndexDims(NumValues(), indices.size()));
  InitCooIndex(index_shape, indices.data());
  return Status::OK();
}

Status SparseTensor::MakeCooData(const IDataTransfer& data_transfer,
                                 const OrtMemoryInfo& data_location,
                                 size_t values_count, const void* values_data,
                                 gsl::span<const int64_t> indices) {
  ORT_RETURN_IF(IsDataTypeString(), "Use MakeCooStrings");
  auto mutator = MakeCooData(values_count, indices.size());
  if (values_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_index = mutator.Indices();

    Tensor src_values(dst_values.DataType(), dst_values.Shape(), const_cast<void*>(values_data), data_location);
    Tensor src_index(dst_index.DataType(), dst_index.Shape(), const_cast<int64_t*>(indices.data()), data_location);
    ORT_RETURN_IF_ERROR(CopyData(&data_transfer, MakeListConst(src_values, src_index), MakeListNonConst(dst_values, dst_index)));
  }
  return Status::OK();
}

Status SparseTensor::MakeCooStrings(size_t string_count, const char* const* strings,
                                    gsl::span<const int64_t> indices) {
  ORT_RETURN_IF_NOT(IsDataTypeString(), "Expecting data type to be set as string");
  auto mutator = MakeCooData(string_count, indices.size());
  if (string_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_indices = mutator.Indices();
    Tensor src_indices(dst_indices.DataType(), dst_indices.Shape(), const_cast<int64_t*>(indices.data()), Location());
    ORT_RETURN_IF_ERROR(CopyStringsAndIndices(string_count, strings, dst_values, {std::cref(src_indices)}, {std::ref(dst_indices)}));
  }
  return Status::OK();
}

SparseTensor::CooMutator SparseTensor::MakeCooData(size_t values_count, size_t index_count) {
  ORT_ENFORCE(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  const auto num_values = narrow<int64_t>(values_count);
  TensorShape values_shape{num_values};
  TensorShape index_shape(GetCooIndexDims(values_count, index_count));
  if (num_values > 0) {
    const auto data_size = SafeInt<size_t>(values_count) * ml_data_type_->Size();
    const auto index_size = SafeInt<size_t>(index_count) * sizeof(int64_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(narrow<int64_t>(data_size),
                                                                  narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, values_count));
  }
  values_ = Tensor(DataType(), values_shape, p_data_, Location());
  InitCooIndex(index_shape, reinterpret_cast<int64_t*>(IndicesStart(values_.SizeInBytes())));
  return CooMutator(values_, format_data_[0]);
}

SparseTensor::CsrView SparseTensor::AsCsr() const {
  ORT_ENFORCE(Format() == SparseFormat::kCsrc, "Must contain Csr format. Contains: ", Format());
  ORT_ENFORCE(format_data_.size() == 2U, "Expecting two indices. Got: ", format_data_.size());
  return CsrView(format_data_[0], format_data_[1]);
}

Status SparseTensor::ValidateCsrIndices(size_t values_count, size_t inner_size, size_t outer_size) const {
  ORT_RETURN_IF_NOT(dense_shape_.NumDimensions() == 2U, "dense shape must 2-D. Got: ", dense_shape_.NumDimensions());
  ORT_RETURN_IF_NOT((inner_size == 0 && outer_size == 0) || (inner_size > 0 && outer_size > 0),
                    "Inner and Outer indices must either be both zero or non-zero");
  ORT_RETURN_IF_NOT(inner_size == values_count,
                    "Expecting inner index size: ", inner_size, " the same as values size: ", values_count);
  const auto rows = dense_shape_.GetDims()[0];
  ORT_RETURN_IF_NOT(outer_size == 0 || outer_size == static_cast<size_t>(rows + 1),
                    "Outer index count must be rows + 1 or zero. Got: ", outer_size, " rows: ", rows);
  return Status::OK();
}

void SparseTensor::InitCsrIndices(size_t inner_size, const int64_t* inner, size_t outer_size, const int64_t* outer) {
  TensorShape inner_shape{static_cast<int64_t>(inner_size)};
  TensorShape outer_shape{static_cast<int64_t>(outer_size)};
  auto index_type = DataTypeImpl::GetType<int64_t>();
  format_data_.resize(2);
  format_data_[0] = Tensor(index_type, inner_shape, const_cast<int64_t*>(inner), Location());
  format_data_[1] = Tensor(index_type, outer_shape, const_cast<int64_t*>(outer), Location());

  format_ = SparseFormat::kCsrc;
}

Status SparseTensor::UseCsrIndices(gsl::span<int64_t> inner_index, gsl::span<int64_t> outer_index) {
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "This method does not expect allocator to be set");
  ORT_RETURN_IF_NOT(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_RETURN_IF_ERROR(ValidateCsrIndices(NumValues(), inner_index.size(), outer_index.size()));
  InitCsrIndices(inner_index.size(), inner_index.data(), outer_index.size(), outer_index.data());
  return Status::OK();
}

Status SparseTensor::MakeCsrData(const IDataTransfer& data_transfer, const OrtMemoryInfo& data_location,
                                 size_t values_count, const void* values_data,
                                 gsl::span<const int64_t> inner_index, gsl::span<const int64_t> outer_index) {
  ORT_RETURN_IF(IsDataTypeString(), "Use MakeCsrStrings");
  auto mutator = MakeCsrData(values_count, inner_index.size(), outer_index.size());
  if (values_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_inner = mutator.Inner();
    auto& dst_outer = mutator.Outer();

    Tensor src_values(dst_values.DataType(), dst_values.Shape(), const_cast<void*>(values_data), data_location);
    Tensor src_inner(dst_inner.DataType(), dst_inner.Shape(), const_cast<int64_t*>(inner_index.data()), data_location);
    Tensor src_outer(dst_outer.DataType(), dst_outer.Shape(), const_cast<int64_t*>(outer_index.data()), data_location);
    ORT_RETURN_IF_ERROR(CopyData(&data_transfer, MakeListConst(src_values, src_inner, src_outer),
                                 MakeListNonConst(dst_values, dst_inner, dst_outer)));
  }
  return Status::OK();
}

Status SparseTensor::MakeCsrStrings(size_t string_count, const char* const* strings,
                                    gsl::span<const int64_t> inner_index, gsl::span<const int64_t> outer_index) {
  ORT_RETURN_IF_NOT(IsDataTypeString(), "Expecting data type to be set as string");
  auto mutator = MakeCsrData(string_count, inner_index.size(), outer_index.size());
  if (string_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_inner = mutator.Inner();
    auto& dst_outer = mutator.Outer();
    Tensor src_inner(dst_inner.DataType(), dst_inner.Shape(), const_cast<int64_t*>(inner_index.data()), Location());
    Tensor src_outer(dst_outer.DataType(), dst_outer.Shape(), const_cast<int64_t*>(outer_index.data()), Location());
    ORT_RETURN_IF_ERROR(CopyStringsAndIndices(string_count, strings, dst_values,
                                              MakeListConst(src_inner, src_outer),
                                              MakeListNonConst(dst_inner, dst_outer)));
  }
  return Status::OK();
}

SparseTensor::CsrMutator SparseTensor::MakeCsrData(size_t values_count,
                                                   size_t inner_index_count,
                                                   size_t outer_index_count) {
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  ORT_ENFORCE(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_THROW_IF_ERROR(ValidateCsrIndices(values_count, inner_index_count, outer_index_count));

  if (values_count > 0) {
    const auto data_size = SafeInt<size_t>(values_count) * ml_data_type_->Size();
    const auto index_size = (SafeInt<size_t>(inner_index_count) + outer_index_count) * sizeof(int64_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(narrow<int64_t>(data_size),
                                                                  narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, values_count));
  }

  const auto num_values = narrow<int64_t>(values_count);
  values_ = Tensor(DataType(), {num_values}, p_data_, Location());

  auto* inner_index_start = reinterpret_cast<int64_t*>(IndicesStart(values_.SizeInBytes()));
  InitCsrIndices(inner_index_count, inner_index_start, outer_index_count, inner_index_start + inner_index_count);
  return CsrMutator(values_, format_data_[0], format_data_[1]);
}

SparseTensor::BlockSparseView SparseTensor::AsBlockSparse() const {
  ORT_ENFORCE(Format() == SparseFormat::kBlockSparse, "Must contain BlockSparse format. Got: ", Format());
  ORT_ENFORCE(format_data_.size() == 1U, "Expecting one index. Got: ", format_data_.size());
  return BlockSparseView(format_data_[0]);
}

Status SparseTensor::ValidateBlockSparseShapes(const TensorShape& values_shape, const TensorShape& indices_shape) const {
  if (values_shape.Size() > 0) {
    ORT_RETURN_IF_NOT(values_shape.NumDimensions() >= 3,
                      "Expecting to have at lest 3-D shape. Got:", values_shape.NumDimensions());
    ORT_RETURN_IF_NOT(indices_shape.NumDimensions() == 2,
                      "Expecting indices to have 2-D shape . Got: ", indices_shape.NumDimensions());
    ORT_RETURN_IF_NOT(indices_shape.GetDims()[0] == 2, "Indices shape must have dim[0] == 2");
    const auto values_blocks = values_shape.SizeFromDimension(2);
    const auto index_blocks = indices_shape.Size() / 2;  // Two integers per block
    ORT_RETURN_IF_NOT(values_blocks == index_blocks,
                      "Expecting index blocks: ", index_blocks, " to be equal to values blocks: ", values_blocks);
  } else {
    ORT_RETURN_IF_NOT(values_shape.GetDims().size() == 1, "Expecting fully sparse tensors to have value shape {0}");
    ORT_RETURN_IF_NOT(indices_shape.GetDims().size() == 1, "Expecting fully sparse tensors to have indices shape {0}");
  }
  return Status::OK();
}

void SparseTensor::InitBlockSparseIndices(const TensorShape& indices_shape, int32_t* indices_data) {
  format_data_.resize(1);
  format_data_[0] = Tensor(DataTypeImpl::GetType<int32_t>(), indices_shape,
                           indices_data, Location());
  format_ = SparseFormat::kBlockSparse;
}

Status SparseTensor::UseBlockSparseIndices(const TensorShape& indices_shape, int32_t* indices_data) {
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Not expecting an allocator set");
  ORT_RETURN_IF_NOT(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_RETURN_IF_ERROR(ValidateBlockSparseShapes(Values().Shape(), indices_shape));
  InitBlockSparseIndices(indices_shape, indices_data);
  return Status::OK();
}

Status SparseTensor::MakeBlockSparseData(const IDataTransfer& data_transfer, const OrtMemoryInfo& data_location,
                                         const TensorShape& values_shape, const void* values_data,
                                         const TensorShape& indices_shape, const int32_t* indices_data) {
  ORT_RETURN_IF(IsDataTypeString(), "Use MakeBlockSparseStrings");
  auto mutator = MakeBlockSparseData(values_shape, indices_shape);
  if (values_shape.Size() > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_indices = mutator.Indices();
    Tensor src_values(dst_values.DataType(), dst_values.Shape(), const_cast<void*>(values_data), data_location);
    Tensor src_index(dst_indices.DataType(), dst_indices.Shape(), const_cast<int32_t*>(indices_data), data_location);
    ORT_RETURN_IF_ERROR(CopyData(&data_transfer, MakeListConst(src_values, src_index), MakeListNonConst(dst_values, dst_indices)));
  }
  return Status::OK();
}

Status SparseTensor::MakeBlockSparseStrings(const TensorShape& values_shape, const char* const* strings,
                                            const TensorShape& indices_shape, const int32_t* indices_data) {
  ORT_RETURN_IF_NOT(IsDataTypeString(), "Expecting data type to be set as string");
  auto mutator = MakeBlockSparseData(values_shape, indices_shape);
  auto string_count = narrow<size_t>(values_shape.Size());
  if (string_count > 0) {
    auto& dst_values = mutator.Values();
    auto& dst_indices = mutator.Indices();
    Tensor src_indices(dst_indices.DataType(), dst_indices.Shape(), const_cast<int32_t*>(indices_data), Location());
    ORT_RETURN_IF_ERROR(CopyStringsAndIndices(string_count, strings, dst_values, {std::cref(src_indices)}, {std::ref(dst_indices)}));
  }
  return Status::OK();
}

SparseTensor::BlockSparseMutator SparseTensor::MakeBlockSparseData(const TensorShape& values_shape, const TensorShape& indices_shape) {
  ORT_ENFORCE(allocator_ != nullptr, "This method should follow a call to constructor that supplies the allocator");
  ORT_ENFORCE(Format() == SparseFormat::kUndefined, "Sparse format must not be set. Already contains format: ", Format());
  ORT_THROW_IF_ERROR(ValidateBlockSparseShapes(values_shape, indices_shape));
  if (values_shape.Size() > 0) {
    const auto data_size = SafeInt<int64_t>(values_shape.Size()) * ml_data_type_->Size();
    const auto index_size = SafeInt<int64_t>(indices_shape.Size()) * sizeof(int32_t);
    const auto required_buffer_size = CalculateRequiredBufferSize(narrow<int64_t>(data_size),
                                                                  narrow<int64_t>(index_size));
    ORT_THROW_IF_ERROR(AllocateBuffer(required_buffer_size, static_cast<size_t>(data_size / ml_data_type_->Size())));
  }

  values_ = Tensor(DataType(), values_shape, p_data_, Location());
  InitBlockSparseIndices(indices_shape, reinterpret_cast<int32_t*>(IndicesStart(values_.SizeInBytes())));
  return BlockSparseMutator(values_, format_data_[0]);
}

Status SparseTensor::Copy(const DataTransferManager& data_transfer_manager, SparseTensor& dst_tensor) const {
  const IDataTransfer* data_transfer = data_transfer_manager.GetDataTransfer(Location().device,
                                                                             dst_tensor.Location().device);
  ORT_RETURN_IF_NOT(data_transfer != nullptr, "Unable to find a data transfer for copying from device type: ",
                    Location().device.Type(), " to device type: ", dst_tensor.Location().device.Type());

  return Copy(*data_transfer, dst_tensor);
}

Status SparseTensor::Copy(const IDataTransfer& data_transfer, SparseTensor& dst_tensor) const {
  // Do not copy same destination
  if (this == &dst_tensor) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(format_ != SparseFormat::kUndefined, "This instance should not be empty");
  const bool is_string = IsDataTypeString();

  ORT_RETURN_IF_NOT(dst_tensor.Format() == SparseFormat::kUndefined, "Destination should be empty");
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have a CPU allocator set");
  ORT_RETURN_IF_NOT((!is_string || dst_tensor.Location().device.Type() == OrtDevice::CPU),
                    "X-device copy of strings not supported");
  ORT_RETURN_IF_NOT(dst_tensor.DataType() == DataType(), "Src and Dst must be of the same type");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() == dense_shape_.Size(), "Must have the same shape");

  const auto required_buffer_size = RequiredAllocationSize();
  SparseTensor result(DataType(), DenseShape(), dst_tensor.allocator_);
  ORT_RETURN_IF_ERROR(result.AllocateBuffer(required_buffer_size, NumValues()));

  // Prepare result Tensor on top of the new buffer
  Tensor result_values(DataType(), Values().Shape(), result.p_data_, result.Location());

  // Setup indices Tensors on top of the new buffer
  const auto values_bytes = values_.SizeInBytes();
  auto* const dst_index_start = reinterpret_cast<int8_t*>(result.IndicesStart(values_bytes));
  result.format_data_.resize(format_data_.size());
  SafeInt<int64_t> index_bytes = 0;
  for (size_t i = 0, size = format_data_.size(); i < size; ++i) {
    const auto& src_idx = format_data_[i];
    result.format_data_[i] = Tensor(src_idx.DataType(), src_idx.Shape(),
                                    dst_index_start + static_cast<int64_t>(index_bytes),
                                    result.Location());
    index_bytes += src_idx.SizeInBytes();
  }

  if (Values().Shape().Size() > 0) {
    // This instance may either have a contigious buffer which we can copy in one shot
    // or it can point to users buffers, in which case we have to copy each buffer individually
    // strings can not be memcpyed albeit always on CPU.
    if (p_data_ != nullptr) {
      if (is_string) {
        CopyStrings(Values(), result_values);
        // We are on CPU, copy indices
        memcpy(dst_index_start, IndicesStart(values_bytes), static_cast<size_t>(index_bytes));
      } else {
        // Copy everything at once, setup artificial tensors
        // that describe the whole buffer
        auto bytes_type = DataTypeImpl::GetType<uint8_t>();
        TensorShape buffer_shape{required_buffer_size};
        Tensor src(bytes_type, buffer_shape, p_data_, Location());
        Tensor dst(bytes_type, buffer_shape, result.p_data_, result.Location());
        ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(src, dst));
      }
    } else {
      // non-contiguos buffer
      if (is_string) {
        CopyStrings(Values(), result_values);
      } else {
        ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Values(), result_values));
      }
      // Copy indices
      for (size_t i = 0, size = format_data_.size(); i < size; ++i) {
        ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(format_data_[i], result.format_data_[i]));
      }
    }
  }

  result.values_ = std::move(result_values);
  result.format_ = Format();

  dst_tensor = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(DISABLE_SPARSE_TENSORS)
