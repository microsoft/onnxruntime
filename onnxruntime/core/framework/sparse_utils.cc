// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_utils.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_cooformat_rep.h"
#include "core/framework/sparse_csrcformat_rep.h"

#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace sparse_utils {

// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

template <typename T>
inline void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<T*>(dst)[dst_index] = reinterpret_cast<const T*>(src)[src_index];
}

template <class T>
void ScanAndRecordCsr(gsl::span<const T> src_span, int64_t cols,
                      std::vector<int64_t>& inner, std::vector<int64_t>& outer,
                      std::vector<T>& values) {
  int64_t row = 0;
  int64_t index = 0;
  outer.push_back(0);
  std::for_each(src_span.cbegin(), src_span.cend(),
                [&](auto v) mutable {
                  auto cur_row = index / cols;
                  if (cur_row != row) {
                    outer.push_back(static_cast<int64_t>(inner.size()));
                    row = cur_row;
                  }
                  if (v != T{0}) {
                    auto cur_col = index - cur_row * cols;
                    inner.push_back(cur_col);
                    values.push_back(v);
                  }
                  ++index;
                });
  outer.push_back(static_cast<int64_t>(inner.size()));
}

common::Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                                      SparseTensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const auto dense_elements = static_cast<size_t>(src.Shape().Size());
  const auto element_size = src.DataType()->Size();
  gsl::span<const uint8_t> src_span;
  Tensor src_cpu;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor t(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
    src_cpu = std::move(t);
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src_cpu.DataRaw()), src_cpu.SizeInBytes());
  } else {
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src.DataRaw()), src.SizeInBytes());
  }

  const auto rows = src_dims[0];
  const auto cols = src_dims[1];

  std::vector<int64_t> inner_indicies;
  inner_indicies.reserve(dense_elements / 2);
  std::vector<int64_t> outer_indices;
  outer_indices.reserve(rows + 1);

  Tensor nnz_tensor;
  std::vector<uint8_t> values_8;
  std::vector<uint16_t> values_16;
  std::vector<uint32_t> values_32;
  std::vector<uint64_t> values_64;

  switch (element_size) {
    case sizeof(uint8_t): {
      values_8.reserve(dense_elements / 2);
      ScanAndRecordCsr(src_span, cols, inner_indicies, outer_indices, values_8);
      Tensor t(src.DataType(), {static_cast<int64_t>(values_8.size())}, values_8.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint16_t): {
      // MFFloat16 and BFloat16 are handled fine
      values_16.reserve(dense_elements / 2);
      auto span16 = src_span.as_span<const uint16_t>();
      ScanAndRecordCsr(span16, cols, inner_indicies, outer_indices, values_16);
      Tensor t(src.DataType(), {static_cast<int64_t>(values_16.size())}, values_16.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint32_t): {
      values_32.reserve(dense_elements / 2);
      auto span32 = src_span.as_span<const uint32_t>();
      ScanAndRecordCsr(span32, cols, inner_indicies, outer_indices, values_32);
      Tensor t(src.DataType(), {static_cast<int64_t>(values_32.size())}, values_32.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint64_t): {
      values_64.reserve(dense_elements / 2);
      auto span64 = src_span.as_span<const uint64_t>();
      ScanAndRecordCsr(span64, cols, inner_indicies, outer_indices, values_64);
      Tensor t(src.DataType(), {static_cast<int64_t>(values_64.size())}, values_64.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
  }

  assert(outer_indices.size() == static_cast<size_t>(rows + 1));
  const auto nnz = inner_indicies.size();
  const size_t outer_size = (nnz > 0) ? outer_indices.size() : 0U;

  SparseTensor dst_tensor(src.DataType(), src.Shape(), dst_allocator);
  SparseCsrcFormatRep* rep;
  ORT_RETURN_IF_ERROR(dst_tensor.RepBuilder<SparseCsrcBuilder>().Create(SparseCsrcFormatRep::kRowMajor,
                                                                        nnz, nnz, outer_size, rep));
  if (nnz > 0) {
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(nnz_tensor, rep->MutableValues()));
    Tensor inner(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(nnz)}, inner_indicies.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(inner, rep->MutableInner()));
    Tensor outer(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(outer_size)},
      outer_indices.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(outer, rep->MutableOuter()));
  }

  dst = std::move(dst_tensor);

  return Status::OK();
}

common::Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                                      Tensor& dst) {
  if (!IsSet(src.FormatFlags(), SparseFormatFlags::kCsrc) ||
      src.GetRep<SparseCsrcFormatRep>()->Major() != SparseCsrcFormatRep::kRowMajor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input must be of CRS format");
  }

  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Support 2-D matrices only");
  }

  const AllocatorPtr& conversion_allocator = (cpu_allocator->Info().device ==
                                              dst_allocator->Info().device)
                                                 ? dst_allocator
                                                 : cpu_allocator;

  Tensor cpu_result(src.DataType(), src.Shape(), conversion_allocator);
  memset(cpu_result.MutableDataRaw(), 0, cpu_result.SizeInBytes());

  if (src.NumValues() > 0) {
    const auto rows = src_dims[0];
    const auto cols = src_dims[1];

    {
      const SparseCsrcFormatRep* rep = src.GetRep<SparseCsrcFormatRep>();
      const auto inner_num = rep->Inner().Shape().Size();
      const auto outer_num = rep->Outer().Shape().Size();
      ORT_ENFORCE(inner_num == src.Values().Shape().Size(), "Expecting inner indecies to be same as nnz. Got: ", inner_num);
      ORT_ENFORCE(outer_num == (rows + 1), "Outer indecies must be M + 1. Got: ", outer_num);
    }

    const auto element_size = src.DataType()->AsPrimitiveDataType()->Size();
    CopyElementFunc copy_func;
    switch (element_size) {
      case sizeof(uint8_t):
        copy_func = CopyElement<uint8_t>;
        break;
      case sizeof(uint16_t):
        copy_func = CopyElement<uint16_t>;
        break;
      case sizeof(uint32_t):
        copy_func = CopyElement<uint32_t>;
        break;
      case sizeof(uint64_t):
        copy_func = CopyElement<uint64_t>;
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
    }

    const SparseCsrcFormatRep* rep = nullptr;
    const void* values = nullptr;
    SparseTensor cpu_src;
    if (src.Location().device != cpu_allocator->Info().device) {
      SparseTensor t(src.DataType(), src.Shape(), cpu_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
      cpu_src = std::move(t);
      rep = cpu_src.GetRep<SparseCsrcFormatRep>();
      values = cpu_src.Values().DataRaw();
    } else {
      rep = src.GetRep<SparseCsrcFormatRep>();
      values = src.Values().DataRaw();
    }

    auto inner_span = rep->Inner().DataAsSpan<int64_t>();
    auto outer_span = rep->Outer().DataAsSpan<int64_t>();
    void* output = cpu_result.MutableDataRaw();

    size_t src_idx = 0;
    size_t inner_idx = 0;
    for (size_t out_i = 1; out_i < outer_span.size(); ++out_i) {
      auto row_size = outer_span[out_i] - outer_span[out_i - 1];
      for (int64_t cnt = 0; cnt < row_size; ++cnt, ++inner_idx) {
        assert(inner_idx < inner_span.size());
        auto col = inner_span[inner_idx];
        auto dst_idx = (out_i - 1) * cols + col;
        copy_func(output, values, dst_idx, src_idx);
      }
    }
  }

  if (cpu_result.Location().device != dst_allocator->Info().device) {
    Tensor dest_tensor(src.DataType(), src.Shape(), dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, dest_tensor));
    dst = std::move(dest_tensor);
  } else {
    dst = std::move(cpu_result);
  }

  return Status::OK();
}

template <typename T>
void ScanAndRecordCoo(gsl::span<const T> src_span, 
                      int64_t cols,
                      bool linear,
                      std::vector<int64_t>& indices,
                      std::vector<T>& values) {
  int64_t index = 0;
  std::for_each(src_span.cbegin(), src_span.cend(),
                [&](T v) mutable {
                  if (v != T{0}) {
                    values.push_back(v);
                    if (linear) {
                      indices.push_back(index);
                    } else {
                      auto row = index / cols;
                      auto col = index - row * cols;
                      indices.push_back(row);
                      indices.push_back(col);
                    }
                  }
                  ++index;
                });
}

Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src,
                              const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_index, SparseTensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const auto dense_elements = static_cast<size_t>(src.Shape().Size());
  const auto element_size = src.DataType()->Size();
  gsl::span<const uint8_t> src_span;
  Tensor src_cpu;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor t(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
    src_cpu = std::move(t);
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src_cpu.DataRaw()), src_cpu.SizeInBytes());
  } else {
    src_span = gsl::make_span(reinterpret_cast<const uint8_t*>(src.DataRaw()), src.SizeInBytes());
  }

  const auto cols = src_dims[1];
  std::vector<uint8_t> values_8;
  std::vector<uint16_t> values_16;
  std::vector<uint32_t> values_32;
  std::vector<uint64_t> values_64;
  Tensor nnz_tensor;

  std::vector<int64_t> gathered_indicies;
  gathered_indicies.reserve(dense_elements / 2);

  switch (element_size) {
    case sizeof(uint8_t): {
      values_8.reserve(dense_elements / 2);
      ScanAndRecordCoo(src_span, cols, linear_index, gathered_indicies, values_8);
      Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_8.size())},
               values_8.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint16_t): {
      values_16.reserve(dense_elements / 2);
      // MFFloat16 and BFloat16 are handled fine
      auto span16 = src_span.as_span<const uint16_t>();
      ScanAndRecordCoo(span16, cols, linear_index, gathered_indicies, values_16);
      Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_16.size())},
               values_16.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint32_t): {
      values_32.reserve(dense_elements / 2);
      auto span32 = src_span.as_span<const uint32_t>();
      ScanAndRecordCoo(span32, cols, linear_index, gathered_indicies, values_32);
      Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_32.size())},
               values_32.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    case sizeof(uint64_t): {
      values_64.reserve(dense_elements / 2);
      auto span64 = src_span.as_span<const uint64_t>();
      ScanAndRecordCoo(span64, cols, linear_index, gathered_indicies, values_64);
      Tensor t(src.DataType(), TensorShape{static_cast<int64_t>(values_64.size())},
               values_32.data(), cpu_allocator->Info());
      nnz_tensor = std::move(t);
    } break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
  }

  const auto nnz = (linear_index) ? gathered_indicies.size() : gathered_indicies.size() / 2;
  assert(static_cast<int64_t>(nnz) == nnz_tensor.Shape().Size());

  SparseTensor dst_result(src.DataType(), src.Shape(), dst_allocator);
  SparseCooFormatRep* rep;
  ORT_RETURN_IF_ERROR(dst_result.RepBuilder<SparseCooBuilder>().Create(linear_index, nnz, rep));
  if (nnz > 0) {
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(nnz_tensor, rep->MutableValues()));
    Tensor indices_tensor(DataTypeImpl::GetType<int64_t>(), rep->Indices().Shape(), gathered_indicies.data(), cpu_allocator->Info());
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(indices_tensor, rep->MutableIndices()));
  }

  dst = std::move(dst_result);

  return Status::OK();
}

Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                              const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator, Tensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const AllocatorPtr& conversion_allocator = (cpu_allocator->Info().device == dst_allocator->Info().device) ? dst_allocator : cpu_allocator;
  Tensor cpu_result(src.DataType(), src.Shape(), conversion_allocator);
  memset(cpu_result.MutableDataRaw(), 0, cpu_result.SizeInBytes());

  if (src.NumValues() > 0) {
    const void* values = nullptr;
    const int64_t* indices = nullptr;
    const auto num_values = src.Values().Shape().Size();
    const auto num_indices = src.GetRep<SparseCooFormatRep>()->Indices().Shape().Size();
    ORT_RETURN_IF_NOT((num_values == num_indices || 2 * num_values == num_indices), "Invalid indices number");
    SparseTensor src_cpu;
    if (src.Location().device != cpu_allocator->Info().device) {
      SparseTensor t(src.DataType(), src.Shape(), cpu_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
      src_cpu = std::move(t);
      values = src_cpu.Values().DataRaw();
      const auto* rep = src_cpu.GetRep<SparseCooFormatRep>();
      indices = rep->Indices().Data<int64_t>();
    } else {
      values = src.Values().DataRaw();
      const auto* rep = src.GetRep<SparseCooFormatRep>();
      indices = rep->Indices().Data<int64_t>();
    }
    const auto element_size = src.DataType()->AsPrimitiveDataType()->Size();
    void* output = cpu_result.MutableDataRaw();

    CopyElementFunc copy_func = nullptr;
    switch (element_size) {
      case sizeof(uint8_t):
        copy_func = CopyElement<uint8_t>;
        break;
      case sizeof(uint16_t): {
        copy_func = CopyElement<uint16_t>;
      } break;
      case sizeof(uint32_t): {
        copy_func = CopyElement<uint32_t>;
      } break;
      case sizeof(uint64_t): {
        copy_func = CopyElement<uint64_t>;
      } break;
      default:
        assert(false);
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
    }

    const auto dense_size = src.Shape().Size();
    // Linear index
    if (num_indices == num_values) {
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto dst_idx = indices[src_idx];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    } else {
      const auto cols = src_dims[1];
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto tuple_idx = src_idx * 2;
        auto dst_idx = indices[tuple_idx] * cols + indices[tuple_idx + 1];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    }
  }

  if (conversion_allocator->Info().device == dst_allocator->Info().device) {
    dst = std::move(cpu_result);
  } else {
    Tensor t(src.DataType(), src.Shape(), dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, t));
    dst = std::move(t);
  }

  return Status::OK();
}

}  // namespace sparse_utils
}  // namespace onnxruntime