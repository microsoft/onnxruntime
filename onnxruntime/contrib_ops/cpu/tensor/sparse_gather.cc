// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/data_transfer_manager.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {
class Gather final : public OpKernel {
 public:
  explicit Gather(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext*) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<element_type_lists::All>()),
    Gather);

Status Gather::Compute(OpKernelContext* ctx) const {
  const SparseTensor& input_tensor = *ctx->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(input_tensor.Format() == SparseFormat::kCoo, "Support COO format");
  const Tensor& indices_tensor = *ctx->Input<Tensor>(1);

  auto& output_tensor = *ctx->OutputSparse(0, input_tensor.DenseShape());
  if (input_tensor.NumValues() == 0 || indices_tensor.Shape().Size() == 0) {
    if (input_tensor.NumValues() > 0) {
      return input_tensor.Copy(Info().GetDataTransferManager(), Info().GetKernelDef().ExecQueueId(), output_tensor);
    } else {
      output_tensor.MakeCooData(0, 0);
    }
    return Status::OK();
  }

  gsl::span<const int64_t> input_indices_span;
  std::vector<int64_t> one_d_indices;
  const auto& coo_indices = input_tensor.AsCoo().Indices();
  if (coo_indices.Shape().NumDimensions() > 1) {
    ORT_ENFORCE(input_tensor.DenseShape().NumDimensions() == 2,
                "Invalid sparse tensor. Only 2-D sparse tensors can have 2-D COO indices");
    const auto cols = input_tensor.DenseShape().GetDims()[1];
    const auto two_d_indices_span = coo_indices.DataAsSpan<int64_t>();
    one_d_indices.resize(two_d_indices_span.size() / 2);
    auto one_d_indices_span = gsl::make_span(one_d_indices);
    sparse_utils::Convert2DCooIndicesTo1D(cols, two_d_indices_span, one_d_indices_span);
    input_indices_span = one_d_indices_span;
  } else {
    input_indices_span = coo_indices.DataAsSpan<int64_t>();
  }

  auto gather_indices_span = indices_tensor.DataAsSpan<int64_t>();
  std::vector<int64_t> gather_sorted(gather_indices_span.cbegin(), gather_indices_span.cend());
  std::sort(gather_sorted.begin(), gather_sorted.end());
  std::vector<size_t> matched_indices_positions;
  matched_indices_positions.reserve(gather_sorted.size());

  size_t inp_ind = 0;
  size_t g_ind = 0;

  const auto inp_ind_size = input_indices_span.size();
  const auto g_ind_size = gather_sorted.size();
  while (inp_ind < inp_ind_size && g_ind < g_ind_size) {
    auto a_val = input_indices_span[inp_ind];
    auto b_val = gather_sorted[g_ind];
    if (a_val == b_val) {
      matched_indices_positions.push_back(inp_ind);
      inp_ind++;
      g_ind++;
    } else if (a_val < b_val) {
      inp_ind++;
    } else {
      g_ind++;
    }
  }

  if (!matched_indices_positions.empty()) {
    sparse_utils::CopyElementFunc copy_func;
    if (input_tensor.IsDataTypeString()) {
      copy_func = sparse_utils::CopyElementAligned<std::string>;
    } else {
      const size_t element_size = input_tensor.DataType()->AsPrimitiveDataType()->Size();
      switch (element_size) {
        case 1: {
          copy_func = sparse_utils::CopyElementAligned<uint8_t>;
          break;
        }
        case 2: {
          copy_func = sparse_utils::CopyElementAligned<uint16_t>;
          break;
        }
        case 4: {
          copy_func = sparse_utils::CopyElementAligned<uint32_t>;
          break;
        }
        case 8: {
          copy_func = sparse_utils::CopyElementAligned<uint64_t>;
          break;
        }
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Element_size of: ", element_size,
                                 " is not supported.", " data_type: ", input_tensor.GetElementType());
      }
    }
    const auto output_elements = matched_indices_positions.size();
    auto coo_mutator = output_tensor.MakeCooData(output_elements, output_elements);
    const auto* input_values = input_tensor.Values().DataRaw();
    auto* output_values = coo_mutator.Values().MutableDataRaw();
    auto* output_indices = coo_mutator.Indices().MutableData<int64_t>();
    int64_t dst_idx = 0;
    for (auto ind : matched_indices_positions) {
      copy_func(output_values, input_values, dst_idx, ind);
      ++dst_idx;
      *output_indices++ = input_indices_span[ind];
    }
  } else {
    ORT_UNUSED_PARAMETER(output_tensor.MakeCooData(0, 0));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // DISABLE_SPARSE_TENSORS