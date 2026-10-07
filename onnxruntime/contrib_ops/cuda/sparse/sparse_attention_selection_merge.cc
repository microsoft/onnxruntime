#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/sparse/sparse_attention_selection_merge_common.h"
#include "contrib_ops/cuda/sparse/sparse_attention_selection_merge_impl.h"

namespace onnxruntime::contrib::cuda {

class SparseAttentionSelectionMerge final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit SparseAttentionSelectionMerge(const OpKernelInfo& info)
      : CudaKernel(info), capacity_(selection_merge::ReadCapacity(info)) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* base = context->Input<Tensor>(0);
    const auto* counts = context->Input<Tensor>(1);
    const auto* rows = context->Input<Tensor>(2);
    const auto* starts = context->Input<Tensor>(3);
    const auto* ends = context->Input<Tensor>(4);
    ORT_RETURN_IF(context->Input<Tensor>(5) || context->Input<Tensor>(6),
                  "append_range does not accept additional index inputs");
    selection_merge::Dimensions dimensions;
    ORT_RETURN_IF_ERROR(selection_merge::Validate(base, counts, rows, starts, ends, capacity_, dimensions));
    auto* output = context->Output(0, {dimensions.queries, capacity_});
    auto* output_counts = context->Output(1, {dimensions.queries});
    auto* status = context->Output(2, {dimensions.queries});
    if (dimensions.queries == 0) return Status::OK();
    auto workspace = GetScratchBuffer<int32_t>(
        static_cast<size_t>(dimensions.queries) * dimensions.hash_capacity * 2, GetComputeStream(context));
    CUDA_RETURN_IF_ERROR(LaunchSparseAttentionSelectionMerge(
        Stream(context), base->Data<int32_t>(), counts->Data<int32_t>(), rows->Data<int32_t>(),
        starts->Data<int32_t>(), ends->Data<int32_t>(), output->MutableData<int32_t>(),
        output_counts->MutableData<int32_t>(), status->MutableData<int32_t>(), workspace.get(),
        dimensions.rows, dimensions.capacity, dimensions.queries, static_cast<int32_t>(capacity_),
        dimensions.hash_capacity));
    return Status::OK();
  }

 private:
  int64_t capacity_;
};

ONNX_OPERATOR_KERNEL_EX(SparseAttentionSelectionMerge, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
                        SparseAttentionSelectionMerge);

}  // namespace onnxruntime::contrib::cuda