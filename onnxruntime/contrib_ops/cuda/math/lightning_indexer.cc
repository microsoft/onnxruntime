// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/lightning_indexer.h"

#include <algorithm>

#include "contrib_ops/cuda/math/lightning_indexer_impl.h"
#include "core/common/inlined_containers.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

// Smallest scoring extent worth clamping to.  Below this the GEMM is small either way, and
// a floor keeps cuBLAS from re-picking a kernel for every slightly different `m`.
constexpr int64_t kMinScoreRows = 256;

// Narrow the scoring GEMM to the cache rows this step can actually reach.
//
// `capacity` is `max_seq_len / ratio`, fixed at export time, while the rows that hold live
// data grow with the sequence.  On a 256K-capable export serving a 4K prompt fewer than 1%
// of them are live, and the select kernel throws the rest away, so scoring them is pure
// loss -- and it is the single largest kernel in prefill.
//
// The bound has to reach the host because cuBLAS takes `m` there, and `past_lens` lives on
// the device.  One synchronised copy per node buys back two orders of magnitude of GEMM, so
// the trade is heavily favourable. During graph capture the live device value cannot be
// used because replay would freeze it. The engine may instead provide a conservative
// request-level maximum through ORT_LIGHTNING_INDEXER_CAPTURE_MAX_PAST; without one,
// capture keeps the full capacity exactly as before.
Status ClampScoreCapacity(OpKernelContext* context, const Tensor& past_lens,
                          LightningIndexerParams& params) {
  cudaStream_t stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));

  int64_t longest = 0;
  if (capture != cudaStreamCaptureStatusNone) {
    longest = ParseEnvironmentVariableWithDefault<int64_t>(
        "ORT_LIGHTNING_INDEXER_CAPTURE_MAX_PAST", 0);
    if (longest <= 0) {
      return Status::OK();
    }
  } else {
    const int batch = params.batch;
    InlinedVector<int64_t> lens(static_cast<size_t>(batch));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(lens.data(), past_lens.Data<int64_t>(),
                                         sizeof(int64_t) * batch, cudaMemcpyDeviceToHost,
                                         stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    for (int b = 0; b < batch; ++b) {
      longest = std::max(longest, lens[b]);
    }
  }
  // The last query of the step sees the most, and one spare row keeps the bound safe
  // against any rounding in the select kernel's own arithmetic.
  int64_t needed =
      LightningIndexerVisibleRows(longest, params.seq_len - 1, params.ratio) + 1;

  // Round up to a power of two so that a prefill, whose bound creeps up by a few rows on
  // every chunk, presents cuBLAS with a handful of distinct `m` values rather than one per
  // chunk.  Costs at most 2x the minimum work and never exceeds the capacity.
  int64_t rounded = kMinScoreRows;
  while (rounded < needed) rounded <<= 1;

  params.score_capacity = static_cast<int>(std::clamp<int64_t>(rounded, 1, params.capacity));
  return Status::OK();
}

}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      LightningIndexer,                                                 \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>())    \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      LightningIndexer<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LightningIndexer<T>::LightningIndexer(const OpKernelInfo& info) : CudaKernel(info) {
  num_heads_ = info.GetAttrOrDefault<int64_t>("num_heads", static_cast<int64_t>(0));
  head_dim_ = info.GetAttrOrDefault<int64_t>("head_dim", static_cast<int64_t>(0));
  rope_head_dim_ = info.GetAttrOrDefault<int64_t>("rope_head_dim", static_cast<int64_t>(0));
  ratio_ = info.GetAttrOrDefault<int64_t>("ratio", static_cast<int64_t>(1));
  topk_ = info.GetAttrOrDefault<int64_t>("topk", static_cast<int64_t>(0));
  max_seq_len_ = info.GetAttrOrDefault<int64_t>("max_seq_len", static_cast<int64_t>(0));
  scale_ = info.GetAttrOrDefault<float>("scale", 1.0f);
  rotate_fp4_ = info.GetAttrOrDefault<int64_t>("rotate_fp4", static_cast<int64_t>(1)) != 0;

  ORT_ENFORCE(num_heads_ > 0, "num_heads must be positive, got ", num_heads_);
  ORT_ENFORCE(head_dim_ > 0, "head_dim must be positive, got ", head_dim_);
  ORT_ENFORCE(rope_head_dim_ >= 0 && rope_head_dim_ <= head_dim_,
              "rope_head_dim must be in [0, head_dim], got ", rope_head_dim_);
  ORT_ENFORCE(ratio_ >= 1, "ratio must be positive, got ", ratio_);
  ORT_ENFORCE(topk_ > 0, "topk must be positive, got ", topk_);
  ORT_ENFORCE(max_seq_len_ > 0, "max_seq_len must be positive, got ", max_seq_len_);
  // The rotation is a Walsh-Hadamard butterfly, and the FP4 grouping is 32 wide.
  ORT_ENFORCE(!rotate_fp4_ || ((head_dim_ & (head_dim_ - 1)) == 0 && head_dim_ % 32 == 0),
              "rotate_fp4 needs head_dim to be a power of two of at least 32, got ", head_dim_);
}

template <typename T>
Status LightningIndexer<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* cos_table = context->Input<Tensor>(1);
  const Tensor* sin_table = context->Input<Tensor>(2);
  const Tensor* rows = context->Input<Tensor>(3);
  const Tensor* first_slot = context->Input<Tensor>(4);
  const Tensor* last_slot = context->Input<Tensor>(5);
  const Tensor* past_cache = context->Input<Tensor>(6);
  const Tensor* weights = context->Input<Tensor>(7);
  const Tensor* past_lens = context->Input<Tensor>(8);

  const auto& q_dims = query->Shape().GetDims();
  if (q_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "query is expected to be [batch, sequence, num_heads * head_dim], "
                           "got rank ",
                           q_dims.size());
  }
  const int64_t batch = q_dims[0];
  const int64_t seq_len = q_dims[1];
  if (q_dims[2] != num_heads_ * head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query must be ",
                           num_heads_ * head_dim_, " wide, got ", q_dims[2]);
  }

  const auto& cache_dims = past_cache->Shape().GetDims();
  if (cache_dims.size() != 3 || cache_dims[0] != batch || cache_dims[2] != head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "past_cache must be [", batch, ", capacity, ", head_dim_, "].");
  }
  const int64_t capacity = cache_dims[1];

  const auto& row_dims = rows->Shape().GetDims();
  if (row_dims.size() != 3 || row_dims[0] != batch || row_dims[2] != head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "rows must be [", batch, ", num_rows, ", head_dim_, "].");
  }
  const int64_t num_rows = row_dims[1];

  const TensorShape slot_shape({batch, 1});
  if (first_slot->Shape() != slot_shape || last_slot->Shape() != slot_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the slot bounds must be shaped [",
                           batch, ", 1].");
  }
  if (weights->Shape().Size() != batch * seq_len * num_heads_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "weights must hold ",
                           batch * seq_len * num_heads_, " elements, got ",
                           weights->Shape().Size());
  }
  const int64_t rope_elems = batch * seq_len * rope_head_dim_;
  if (rope_head_dim_ > 0 &&
      (cos_table->Shape().Size() != rope_elems || sin_table->Shape().Size() != rope_elems)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the rotary slices must hold ",
                           rope_elems, " elements each.");
  }
  if (past_lens->Shape().Size() != batch) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "past_lens must hold ", batch,
                           " elements, got ", past_lens->Shape().Size());
  }
  if (topk_ > capacity) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "topk ", topk_,
                           " exceeds the cache capacity ", capacity, ".");
  }

  Tensor* selection = context->Output(0, TensorShape({batch, seq_len, topk_}));
  Tensor* present_cache = context->Output(1, past_cache->Shape());

  LightningIndexerParams params;
  params.batch = static_cast<int>(batch);
  params.seq_len = static_cast<int>(seq_len);
  params.num_heads = static_cast<int>(num_heads_);
  params.head_dim = static_cast<int>(head_dim_);
  params.rope_head_dim = static_cast<int>(rope_head_dim_);
  params.nope_dim = static_cast<int>(head_dim_ - rope_head_dim_);
  params.num_rows = static_cast<int>(num_rows);
  params.capacity = static_cast<int>(capacity);
  params.score_capacity = static_cast<int>(capacity);
  params.ratio = static_cast<int>(ratio_);
  params.topk = static_cast<int>(topk_);
  params.max_seq_len = static_cast<int>(max_seq_len_);
  params.scale = scale_;
  params.rotate_fp4 = rotate_fp4_;

  ORT_RETURN_IF_ERROR(ClampScoreCapacity(context, *past_lens, params));

  auto query_scratch = GetScratchBuffer<float>(
      static_cast<size_t>(LightningIndexerQueryElems(params)), context->GetComputeStream());
  auto cache_scratch = GetScratchBuffer<float>(
      static_cast<size_t>(LightningIndexerCacheElems(params)), context->GetComputeStream());
  auto score_scratch = GetScratchBuffer<float>(
      static_cast<size_t>(LightningIndexerScoreElems(params)), context->GetComputeStream());
  auto key_scratch = GetScratchBuffer<uint32_t>(
      static_cast<size_t>(LightningIndexerKeyElems(params)), context->GetComputeStream());

  return LaunchLightningIndexer<T>(
      Stream(context), GetCublasHandle(context), GetDeviceProp(), UseTF32(), params,
      query->Data<T>(), cos_table->Data<float>(), sin_table->Data<float>(), rows->Data<T>(),
      first_slot->Data<int64_t>(), last_slot->Data<int64_t>(), past_cache->Data<T>(),
      weights->Data<T>(), past_lens->Data<int64_t>(), selection->MutableData<int64_t>(),
      present_cache->MutableData<T>(), query_scratch.get(), cache_scratch.get(),
      score_scratch.get(), key_scratch.get());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
