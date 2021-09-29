// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/data_transfer_manager.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/tensor.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {
class MakeCooSparse final : public OpKernel {
 public:
  explicit MakeCooSparse(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext*) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    MakeCooSparse,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<element_type_lists::All>())
        .TypeConstraint("T2", BuildKernelDefSparseConstraintsFromTypeList<element_type_lists::All>()),
    MakeCooSparse);

namespace {
Status VerifyFlatIndices(gsl::span<const int64_t> indices_span, int64_t dense_values_count) {
  ORT_RETURN_IF_NOT(std::is_sorted(indices_span.cbegin(), indices_span.cend()),
                    "COO indices must be sorted in ascending order");
  for (auto idx : indices_span) {
    ORT_RETURN_IF_NOT(idx < dense_values_count,
                      "Index: ", idx, " out of bounds. Dense shape count: ", dense_values_count);
  }
  return Status::OK();
}
}

Status MakeCooSparse::Compute(OpKernelContext* ctx) const {
  const Tensor& dense_shape_input = *ctx->Input<Tensor>(0);
  const Tensor& values_input = *ctx->Input<Tensor>(1);
  const Tensor& indices_input = *ctx->Input<Tensor>(2);

  const auto dense_dims_span = dense_shape_input.DataAsSpan<int64_t>();
  TensorShape dense_shape(dense_dims_span.data(), dense_dims_span.size());

  const size_t nnz_values_count = gsl::narrow<size_t>(values_input.Shape().Size());
  const auto dense_values_count = dense_shape.Size();
  ORT_RETURN_IF_NOT(nnz_values_count <= gsl::narrow<size_t>(dense_values_count),
                    "Supplied NNZ values count is greater than dense shape count");

  const auto indices_span = indices_input.DataAsSpan<int64_t>();
  const auto indices_ndims = indices_input.Shape().NumDimensions();
  const auto dense_ndims = dense_shape.NumDimensions();
  ORT_RETURN_IF(indices_ndims > dense_ndims,
                "indices must not have more dimensions than dense_shape");
  if (indices_ndims == 1) {
    VerifyFlatIndices(indices_span, dense_values_count);
  } else if (indices_ndims == 2) {
    // Sometimes we get indices with a shape {nnz, 1} which is really a 1-D indices
    // so we want to check for that.
    if (indices_span.size() == nnz_values_count) {
      VerifyFlatIndices(indices_span, dense_values_count);
    } else {
      ORT_RETURN_IF_NOT(dense_ndims == 2, "Dense shape must be 2-D for 2-D indices");
      ORT_RETURN_IF_NOT(indices_span.size() == nnz_values_count * 2,
                        "Expecting indices have 2x entries as NNZ values");
      const auto rows = dense_shape.GetDims()[0];
      const auto cols = dense_shape.GetDims()[1];
      for (size_t i = 0, limit = indices_span.size(); i < limit; i += 2) {
        ORT_RETURN_IF_NOT(indices_span[i] < rows,
                          "Indices row is out of bounds: ", indices_span[i], " rows: ", rows);
        ORT_RETURN_IF_NOT(indices_span[i + 1] < cols,
                          "Indices col is out of bounds: ", indices_span[i + 1], " cols: ", cols);
      }
      std::vector<int64_t> flat_indices;
      flat_indices.resize(indices_span.size() / 2);
      ORT_RETURN_IF_ERROR(sparse_utils::Convert2DCooIndicesTo1D(cols, indices_span, gsl::make_span(flat_indices)));
      ORT_RETURN_IF_NOT(std::is_sorted(flat_indices.cbegin(), flat_indices.cend()),
                        "COO indices must be sorted in ascending order");
    }
  }

  SparseTensor& output = *ctx->OutputSparse(0, dense_shape);
  const auto& dtm = Info().GetDataTransferManager();
  const auto* data_transfer = dtm.GetDataTransfer(values_input.Location().device, output.Location().device);
  ORT_RETURN_IF(data_transfer == nullptr, "Can not find corresponding data transfer");
  const void* values_data = values_input.DataRaw();
  ORT_RETURN_IF_ERROR(output.MakeCooData(*data_transfer, values_input.Location(), nnz_values_count, values_data,
                                         indices_span));

  return Status::OK();
}

class SparseDecomposeToDense final : public OpKernel {
 public:
  explicit SparseDecomposeToDense(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext*) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    SparseDecomposeToDense,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<element_type_lists::All>())
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<element_type_lists::All>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int64_t>()),
    SparseDecomposeToDense);

Status SparseDecomposeToDense::Compute(OpKernelContext* ctx) const {
  const SparseTensor& input_tensor = *ctx->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(input_tensor.Format() == SparseFormat::kCoo, "Input must be Sparse COO format");

  const auto& dense_shape = input_tensor.DenseShape();
  Tensor& output_shape = *ctx->Output(0, {gsl::narrow<int64_t>(dense_shape.NumDimensions())});
  memcpy(output_shape.MutableDataRaw(), dense_shape.GetDims().data(), output_shape.SizeInBytes());

  const auto& input_values = input_tensor.Values();
  const auto& input_values_shape = input_values.Shape();
  Tensor& output_values = *ctx->Output(1, input_values_shape);
  sparse_utils::CopySparseCpuValues(input_tensor, output_values);

  // XXX: We always output 1-D indices, as the initial examples suggest
  // that the customer prefers to deal with 1-D indices
  const auto& input_indices = input_tensor.AsCoo().Indices();
  const auto& input_indices_shape = input_indices.Shape();
  if (input_indices_shape.NumDimensions() == 1) {
    Tensor& output_indices = *ctx->Output(2, input_indices_shape);
    memcpy(output_indices.MutableDataRaw(), input_indices.DataRaw(), input_indices.SizeInBytes());
  } else if (input_indices_shape.NumDimensions() == 2) {
    ORT_RETURN_IF_NOT(dense_shape.NumDimensions() == 2, "Expecting 2-D Sparse input for 2-D coo indices");
    const auto dst_indices_size = input_indices_shape.Size() / 2;
    ORT_RETURN_IF_NOT(input_values_shape.Size() == dst_indices_size, "Invalid input. 1-D indices size must be equal to that of values");
    Tensor& output_indices = *ctx->Output(2, {dst_indices_size});
    if (dst_indices_size > 0) {
      const auto cols = dense_shape.GetDims()[1];
      sparse_utils::Convert2DCooIndicesTo1D(cols, input_indices.DataAsSpan<int64_t>(),
                                            output_indices.MutableDataAsSpan<int64_t>());
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid COO indices shape: ", input_indices.Shape().ToString());
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
