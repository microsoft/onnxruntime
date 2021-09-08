// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/data_transfer_manager.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/squeeze.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

namespace onnxruntime {
namespace contrib {

class Squeeze final : public OpKernel, public SqueezeBase {
 public:
  explicit Squeeze(const OpKernelInfo& info) : OpKernel(info), SqueezeBase(info) {}

  Status Compute(OpKernelContext*) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<element_type_lists::All>()),
    Squeeze);

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : OpKernel(info), UnsqueezeBase(info) {}
  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    Unsqueeze,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<element_type_lists::All>()),
    Unsqueeze);


Status Squeeze::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(X.Format() == SparseFormat::kCoo, "Only COO format is supported.");
  const TensorShape& X_shape = X.DenseShape();
  const auto input_ndims = X_shape.NumDimensions();

  // Can't squeeze 1-D Sparse tensor
  // returning a copy of it
  if (input_ndims == 1) {
    SparseTensor& Y = *context->OutputSparse(0, X_shape);
    return Info().GetDataTransferManager().CopySparseTensor(X, Y, Info().GetKernelDef().ExecQueueId());
  }

  ORT_RETURN_IF_NOT(input_ndims == 2, "Expecting 2-D sparse tensor on input");

  std::vector<int64_t> axes = ComputeAxes(context, axes_);
  ORT_RETURN_IF_NOT(axes.size() <= 1, "Axes can contain only a single axis for sparse tensors");
  std::vector<int64_t> output_shape = ComputeOutputShape(X_shape, axes);
  // Sparse tensors do not support empty shapes
  if (output_shape.empty()) {
    output_shape.push_back(1);
  }

  SparseTensor& Y = *context->OutputSparse(0, TensorShape(output_shape));
  auto coo_view = X.AsCoo();
  const auto& indices = coo_view.Indices();
  if (indices.Shape().NumDimensions() > 1) {
    auto new_indices_size = gsl::narrow<size_t>(indices.Shape().Size()) / 2;
    auto coo_mutator = Y.MakeCooData(X.NumValues(), new_indices_size);
    sparse_utils::CopySparseCpuValues(X, coo_mutator.Values());
    ORT_RETURN_IF_ERROR(sparse_utils::ConvertIndicesTo1DAndCopy(X, coo_mutator));
  } else {
    sparse_utils::CopyCpuSparseCooTensor(X, Y);
  }
 
  return Status::OK();
}

Status Unsqueeze::Compute(OpKernelContext* ctx) const {
  const auto& input_tensor = *ctx->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(input_tensor.Format() == SparseFormat::kCoo, "Only COO format is supported.");
  ORT_RETURN_IF_NOT(input_tensor.DenseShape().NumDimensions() == 1, "Expected to get 1-D SparseTensor on input");
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, input_tensor.DenseShape(), output_shape));
  auto* output_tensor = ctx->OutputSparse(0, output_shape);
  ORT_RETURN_IF(nullptr == output_tensor, "Failed to get output tensor");
  sparse_utils::CopyCpuSparseCooTensor(input_tensor, *output_tensor);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  //  !defined(DISABLE_SPARSE_TENSORS)
