#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_merge_common.h"
#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer_merge_impl.h"

namespace onnxruntime::contrib::cuda {

class PackedSparseAttentionIndexerMerge final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit PackedSparseAttentionIndexerMerge(const OpKernelInfo& info)
      : CudaKernel(info), capacity_(indexer_merge::ReadCapacity(info)) {
    std::string policy;
    ORT_THROW_IF_ERROR(info.GetAttr("policy_mode", &policy));
    append_indices_ = policy == "append_indices";
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* base = context->Input<Tensor>(0);
    const auto* counts = context->Input<Tensor>(1);
    const auto* rows = context->Input<Tensor>(2);
    const auto* starts = context->Input<Tensor>(3);
    const auto* ends = context->Input<Tensor>(4);
    const auto* additional = context->Input<Tensor>(5);
    const auto* additional_counts = context->Input<Tensor>(6);
    indexer_merge::Dimensions dimensions;
    if (append_indices_) {
      ORT_RETURN_IF(starts || ends, "append_indices does not accept ranges");
      ORT_RETURN_IF_ERROR(indexer_merge::Validate(base, counts, rows, rows, rows, capacity_, dimensions));
      ORT_RETURN_IF(additional == nullptr || additional_counts == nullptr ||
                        additional->Shape().NumDimensions() != 2 ||
                        additional->Shape()[0] != dimensions.queries ||
                        additional_counts->Shape() != rows->Shape() ||
                        additional->Shape()[1] > INT32_MAX - dimensions.capacity,
                    "Additional selections must have shape [N,A] and counts [N], with bounded capacity");
    } else {
      ORT_RETURN_IF(additional || additional_counts, "append_range does not accept additional index inputs");
      ORT_RETURN_IF_ERROR(indexer_merge::Validate(base, counts, rows, starts, ends, capacity_, dimensions));
    }
    auto* output = context->Output(0, {dimensions.queries, capacity_});
    auto* output_counts = context->Output(1, {dimensions.queries});
    auto* status = context->Output(2, {dimensions.queries});
    if (dimensions.queries == 0) return Status::OK();
    auto workspace = GetScratchBuffer<int32_t>(
        static_cast<size_t>(dimensions.queries) * dimensions.hash_capacity * 2, GetComputeStream(context));
    if (append_indices_) {
      CUDA_RETURN_IF_ERROR(LaunchPackedSparseAttentionIndexerMergeIndices(
          Stream(context), base->Data<int32_t>(), counts->Data<int32_t>(), rows->Data<int32_t>(),
          additional->Data<int32_t>(), additional_counts->Data<int32_t>(), output->MutableData<int32_t>(),
          output_counts->MutableData<int32_t>(), status->MutableData<int32_t>(), workspace.get(),
          dimensions.rows, dimensions.capacity, static_cast<int32_t>(additional->Shape()[1]),
          dimensions.queries, static_cast<int32_t>(capacity_), dimensions.hash_capacity));
      return Status::OK();
    }
    CUDA_RETURN_IF_ERROR(LaunchPackedSparseAttentionIndexerMerge(
        Stream(context), base->Data<int32_t>(), counts->Data<int32_t>(), rows->Data<int32_t>(),
        starts->Data<int32_t>(), ends->Data<int32_t>(), output->MutableData<int32_t>(),
        output_counts->MutableData<int32_t>(), status->MutableData<int32_t>(), workspace.get(),
        dimensions.rows, dimensions.capacity, dimensions.queries, static_cast<int32_t>(capacity_),
        dimensions.hash_capacity));
    return Status::OK();
  }

 private:
  int64_t capacity_;
  bool append_indices_;
};

ONNX_OPERATOR_KERNEL_EX(PackedSparseAttentionIndexerMerge, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
                        PackedSparseAttentionIndexerMerge);

}  // namespace onnxruntime::contrib::cuda