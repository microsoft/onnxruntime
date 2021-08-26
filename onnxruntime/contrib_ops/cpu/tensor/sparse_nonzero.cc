// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/sparse_tensor.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace contrib {

class NonZero final : public OpKernel {
 public:
  explicit NonZero(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext*) const override;
 };

ONNX_OPERATOR_KERNEL_EX(
     NonZero,
     kMSDomain,
     1,
     kCpuExecutionProvider,
     KernelDefBuilder()
         .TypeConstraint("T", BuildKernelDefSparseConstraints<float, double, int8_t, uint8_t,
                         int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>())
        .TypeConstraint("T1", BuildKernelDefConstraints<int64_t>()),
    NonZero);

Status NonZero::Compute(OpKernelContext* ctx) const {
  const auto* A = ctx->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(A->Format() == SparseFormat::kCoo, "Implementation supports COO format");
  const auto& A_shape = A->DenseShape();
  const auto dense_dims = A_shape.NumDimensions();
  ORT_RETURN_IF_NOT(dense_dims == 1 || dense_dims == 2, "Expecting 1-D or 2-D tensor");

  // Fully sparse tensor
  if (A->NumValues() == 0) {
    ORT_UNUSED_PARAMETER(ctx->Output(0, {static_cast<int64_t>(dense_dims), 0}));
    return Status::OK();
  }

  const size_t element_size = A->DataType()->AsPrimitiveDataType()->Size();
  sparse_utils::IsZeroFunc is_zero;
  switch (element_size) {
    case 1:
      is_zero = sparse_utils::IsZero<uint8_t>;
      break;
    case 2:
      is_zero = sparse_utils::IsZero<uint16_t>;
      break;
    case 4:
      is_zero = sparse_utils::IsZero<uint32_t>;
      break;
    case 8:
      is_zero = sparse_utils::IsZero<uint64_t>;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             " Element_size of: ",
                             element_size, " is not supported.", " data_type: ", A->GetElementType());
  }

  const auto elements = A->NumValues();
  auto advance = [element_size](const void* start, size_t elements) -> const void* {
    return (reinterpret_cast<const uint8_t*>(start) + elements * element_size);
  };

  std::vector<size_t> ind_ind;
  ind_ind.reserve(elements);
  {
    size_t ind = 0;
    const auto* cbegin = A->Values().DataRaw();
    const auto* const cend = advance(cbegin, elements);
    while (cbegin != cend) {
      if (!is_zero(cbegin)) {
        ind_ind.push_back(ind);
      }
      ++ind;
      cbegin = advance(cbegin, 1U);
    }
  }

  const auto non_zeros = ind_ind.size();
  std::vector<int64_t> output_dims{static_cast<int64_t>(dense_dims),
                                   static_cast<int64_t>(non_zeros)};
  auto* Y = ctx->Output(0, output_dims);
  // All zeros
  if (non_zeros > 0) {
    auto coo_view = A->AsCoo();
    const auto& indices = coo_view.Indices();
    const auto ind_dims = indices.Shape().NumDimensions();
    ORT_RETURN_IF_NOT(ind_dims == 1 || ind_dims == 2, "Expecting indices dims == 1 || 2");
    ORT_RETURN_IF_NOT(dense_dims >= ind_dims, "Expecting dense dims to be GE than indices dims");
    const auto indices_span = indices.DataAsSpan<int64_t>();
    auto* output_data = Y->MutableData<int64_t>();
    if (dense_dims == 1) {  // flat data and indices
      for (auto ind : ind_ind) {
        *output_data++ = indices_span[ind];
      }
    } else {
      // 2-D dense shape
      if (ind_dims == 1) {
        const auto cols = A_shape.GetDims()[1];
        // flat indices to 2-D
        for (auto ind : ind_ind) {
          auto row = indices_span[ind] / cols;
          auto col = indices_span[ind] - cols * row;
          *output_data = row;
          *(output_data + non_zeros) = col;
          ++output_data;
        }
      } else {
        // 2-D indices to 2-D
        for (auto ind : ind_ind) {
          auto row = indices_span[ind * 2];
          auto col = indices_span[ind * 2 + 1];
          *output_data = row;
          *(output_data + non_zeros) = col;
          ++output_data;
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
