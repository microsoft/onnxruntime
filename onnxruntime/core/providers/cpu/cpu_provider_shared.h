// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

namespace contrib {
class LongformerAttentionBase;
class AttentionBase;
namespace transformers {
class BeamSearch;
class WhisperBeamSearch;
class GreedySearch;
class Sampling;
struct BeamSearchParameters;
struct GreedySearchParameters;
struct SamplingParameters;
struct WhisperBeamSearchParameters;
}  // namespace transformers
}  // namespace contrib

struct MLFloat16;

class GatherBase__Prepare;
class ConcatBase_InlinedTensorsVector;
class SliceOp__PrepareForComputeMetadata;  // Directly maps to SliceOp::PrepareForComputeMetadata
class UnsqueezeBase__Prepare;              // Directly maps to UnsqueezeBase::Prepare
class contrib__AdamWOptimizerBase__Prepare;
class contrib__SGDOptimizerV2Base__Prepare;
class UpsampleBase;
class EinsumComputePreprocessor;

namespace EinsumOp {
namespace DeviceHelpers {
using DataCopy = std::function<Status(const Tensor& input, Tensor& output, void* einsum_cuda_assets)>;
using CreateTensor = std::function<std::unique_ptr<Tensor>(const DataTypeImpl* type, const TensorShape& shape, AllocatorPtr allocator)>;
using ZeroBuffer = std::function<Status(Tensor& input, void* einsum_cuda_assets)>;
using Transpose = std::function<Status(const gsl::span<const size_t>& permutation, const Tensor& input,
                                       Tensor& output, const TensorShape* input_shape_override,
                                       void* einsum_cuda_assets)>;

template <typename T>
using MatMul = std::function<Status(const T* input_1_data, const T* input_2_data, T* output_data,
                                    size_t left_stride, size_t right_stride, size_t output_stride,
                                    size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
                                    const void* mlas_backend_config,
                                    void* einsum_cuda_assets)>;

template <typename T>
using ReduceSum = std::function<std::unique_ptr<Tensor>(const Tensor& input, gsl::span<const int64_t> reduce_axes,
                                                        bool keep_dims, AllocatorPtr allocator,
                                                        const TensorShape* input_shape_override,
                                                        concurrency::ThreadPool* tp, void* einsum_cuda_assets)>;
}  // namespace DeviceHelpers
}  // namespace EinsumOp

using PadsVector = InlinedVector<int64_t, kTensorShapeSmallBufferElementsSize * 2>;

struct ProviderHostCPU {
  // From cpu/tensor/gatherbase.h
  virtual Status GatherBase__PrepareForCompute(const GatherBase* p, OpKernelContext* context, GatherBase__Prepare& prepare) = 0;
  // From cpu/tensor/unsqueeze.h
  virtual Status UnsqueezeBase__PrepareCompute(const UnsqueezeBase* p, OpKernelContext* ctx, UnsqueezeBase__Prepare& prepare) = 0;

  // NonMaxSuppresionBase
  virtual Status NonMaxSuppressionBase__PrepareCompute(OpKernelContext* ctx, PrepareContext& pc) = 0;
  virtual Status NonMaxSuppressionBase__GetThresholdsFromInputs(const PrepareContext& pc, int64_t& max_output_boxes_per_class, float& iou_threshold, float& score_threshold) = 0;

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)

  // From cpu/tensor/size.h
  virtual Status Size__Compute(const Size* p, OpKernelContext* context) = 0;
  // From cpu/tensor/scatter_nd.h
  virtual Status ScatterNDBase__ValidateShapes(const TensorShape& input_shape,
                                               const TensorShape& indice_shape,
                                               const TensorShape& update_shape) = 0;
  // From cpu/tensor/padbase.h
  virtual Status PadBase__HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, const TensorShape& output_shape) = 0;

  virtual void PadBase__ComputePads(OpKernelContext& ctx, size_t data_rank, gsl::span<const int64_t> pads_data,
                                    PadsVector& pads) = 0;

  // From cpu/tensor/split.h
  virtual Status SplitBase__PrepareForCompute(const SplitBase* p, const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                              int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                              std::vector<int64_t>& split_sizes) = 0;
  // From cpu/tensor/concatbase.h
  virtual Status ConcatBase__PrepareForCompute(const ConcatBase* p, OpKernelContext* ctx, const ConcatBase_InlinedTensorsVector& input_tensors, Prepare& prepare) = 0;

  // GatherElements
  virtual Status GatherElements__ValidateInputShapes(const TensorShape& input_data_shape, const TensorShape& indices_shape, int64_t axis) = 0;
  // cumsum.cc
  virtual Status cumsum_op__GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) = 0;

  // TileOp
  virtual bool TileOp__IsTileMemcpy(const TensorShape& input_shape, const int64_t* repeats, size_t rank, bool& is_batched_memcpy, size_t& num_of_elements_per_batch, size_t& num_of_copies_per_batch, size_t& num_of_batch_copies) = 0;

  // ROI
  virtual Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) = 0;

  // From onehot.h
  virtual Status ValidateInputs(const Tensor* depth, const Tensor* values) = 0;
  virtual Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis, int64_t& prefix_dim_size, int64_t& suffix_dim_size, TensorShapeVector& output_shape) = 0;

  // From cpu/tensor/slice.h
  virtual Status SliceBase__FlattenOutputDims(gsl::span<const int64_t> input_dimensions, gsl::span<const int64_t> output_dims,
                                              TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                              TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) = 0;

  virtual Status SliceBase__PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                              gsl::span<const int64_t> raw_ends,
                                              gsl::span<const int64_t> raw_axes,
                                              SliceOp__PrepareForComputeMetadata& compute_metadata) = 0;

  virtual Status SliceBase__PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                              gsl::span<const int64_t> raw_ends,
                                              gsl::span<const int64_t> raw_axes,
                                              gsl::span<const int64_t> raw_steps,
                                              SliceOp__PrepareForComputeMetadata& compute_metadata) = 0;
  virtual Status SliceBase__FillVectorsFromInput(const Tensor& start_tensor,
                                                 const Tensor& ends_tensor,
                                                 const Tensor* axes_tensor,
                                                 const Tensor* steps_tensor,
                                                 TensorShapeVector& input_starts,
                                                 TensorShapeVector& input_ends,
                                                 TensorShapeVector& input_axes,
                                                 TensorShapeVector& input_steps) = 0;

  virtual Status Einsum__Compute(const Einsum* p, OpKernelContext* context) = 0;

  // If
  virtual void If__Init(If* p, const OpKernelInfo& info) = 0;
  virtual Status If__Compute(const If* p, OpKernelContext* ctx) = 0;
  virtual Status If__SetupSubgraphExecutionInfo(If* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;

  // Loop
  virtual void Loop__Init(Loop* p, const OpKernelInfo& info) = 0;
  virtual Status Loop__Compute(const Loop* p, OpKernelContext* ctx) = 0;
  virtual Status Loop__SetupSubgraphExecutionInfo(Loop* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;

  // Scan
  virtual void Scan__Init(Scan<8>* p, const OpKernelInfo& info) = 0;
  virtual void Scan__Init(Scan<9>* p, const OpKernelInfo& info) = 0;
  virtual Status Scan__Compute(const Scan<8>* p, OpKernelContext* ctx) = 0;
  virtual Status Scan__Compute(const Scan<9>* p, OpKernelContext* ctx) = 0;
  virtual Status Scan__SetupSubgraphExecutionInfo(Scan<8>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;
  virtual Status Scan__SetupSubgraphExecutionInfo(Scan<9>* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;
  virtual void UpsampleBase__AdjustOutputSizeAsPolicy(const UpsampleBase* p, TensorShapeVector& output_dims,
                                                      gsl::span<const int64_t> input_dims,
                                                      InlinedVector<float>& scales) const = 0;

  virtual Status EinsumTypedComputeProcessor_float_Compute(
      OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp,
      const void* mlas_backend_config, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets,
      const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
      const EinsumOp::DeviceHelpers::MatMul<float>& device_matmul_func,
      const EinsumOp::DeviceHelpers::ReduceSum<float>& device_reduce_sum_func,
      const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func,
      const EinsumOp::DeviceHelpers::ZeroBuffer& device_zero_buffer_func,
      const EinsumOp::DeviceHelpers::CreateTensor& device_create_tensor_func) = 0;

  virtual Status EinsumTypedComputeProcessor_double_Compute(
      OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp,
      const void* mlas_backend_config, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets,
      const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
      const EinsumOp::DeviceHelpers::MatMul<double>& device_matmul_func,
      const EinsumOp::DeviceHelpers::ReduceSum<double>& device_reduce_sum_func,
      const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func,
      const EinsumOp::DeviceHelpers::ZeroBuffer& device_zero_buffer_func,
      const EinsumOp::DeviceHelpers::CreateTensor& device_create_tensor_func) = 0;

  virtual Status EinsumTypedComputeProcessor_MLFloat16_Compute(
      OpKernelContext* context, AllocatorPtr allocator, concurrency::ThreadPool* tp,
      const void* mlas_backend_config, EinsumComputePreprocessor& einsum_compute_preprocessor, void* einsum_cuda_assets,
      const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
      const EinsumOp::DeviceHelpers::MatMul<MLFloat16>& device_matmul_func,
      const EinsumOp::DeviceHelpers::ReduceSum<MLFloat16>& device_reduce_sum_func,
      const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func,
      const EinsumOp::DeviceHelpers::ZeroBuffer& device_zero_buffer_func,
      const EinsumOp::DeviceHelpers::CreateTensor& device_create_tensor_func) = 0;
#ifndef DISABLE_CONTRIB_OPS
  virtual Status embed_layer_norm__CheckInputs(const OpKernelContext* context, bool quantizedVersion) = 0;
  virtual Status bias_gelu_helper__CheckInputs(const OpKernelContext* context) = 0;

  virtual Status LongformerAttentionBase__CheckInputs(const contrib::LongformerAttentionBase* p,
                                                      const TensorShape& input_shape,
                                                      const TensorShape& weights_shape,
                                                      const TensorShape& bias_shape,
                                                      const TensorShape& mask_shape,
                                                      const TensorShape& global_weights_shape,
                                                      const TensorShape& global_bias_shape,
                                                      const TensorShape& global_shape) = 0;

  virtual Status AttentionBase__CheckInputs(const contrib::AttentionBase* p,
                                            const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const Tensor*& mask_index,
                                            const Tensor* past,
                                            const Tensor* attention_bias,
                                            void* parameters,
                                            const int max_threads_per_block,
                                            const Tensor* past_seq_len) = 0;

  virtual Tensor* AttentionBase__GetPresent(const contrib::AttentionBase* p,
                                            OpKernelContext* context,
                                            const Tensor* past,
                                            int batch_size,
                                            int head_size,
                                            int sequence_length,
                                            int& past_sequence_length) = 0;

#if !defined(DISABLE_GENERATION_OPS)
  // BeamSearch
  virtual void BeamSearch__Init(contrib::transformers::BeamSearch* p, const OpKernelInfo& info) = 0;
  virtual Status BeamSearch__Compute(const contrib::transformers::BeamSearch* p, OpKernelContext* ctx) = 0;
  virtual Status BeamSearch__SetupSubgraphExecutionInfo(contrib::transformers::BeamSearch* p,
                                                        const SessionState& session_state,
                                                        const std::string& attribute_name,
                                                        const SessionState& subgraph_session_state) = 0;
  virtual Status WhisperBeamSearch__Compute(const contrib::transformers::WhisperBeamSearch* p, OpKernelContext* ctx) = 0;

  virtual void BeamSearchParameters__ParseFromAttributes(contrib::transformers::BeamSearchParameters* p, const OpKernelInfo& info) = 0;

  virtual void GreedySearchParameters__ParseFromAttributes(contrib::transformers::GreedySearchParameters* p, const OpKernelInfo& info) = 0;

  virtual void SamplingParameters__ParseFromAttributes(contrib::transformers::SamplingParameters* p, const OpKernelInfo& info) = 0;

  virtual void WhisperBeamSearchParameters__ParseFromAttributes(contrib::transformers::WhisperBeamSearchParameters* p, const OpKernelInfo& info) = 0;

  // GreedySearch
  virtual void GreedySearch__Init(contrib::transformers::GreedySearch* p, const OpKernelInfo& info) = 0;
  virtual Status GreedySearch__Compute(const contrib::transformers::GreedySearch* p, OpKernelContext* ctx) = 0;
  virtual Status GreedySearch__SetupSubgraphExecutionInfo(contrib::transformers::GreedySearch* p,
                                                          const SessionState& session_state,
                                                          const std::string& attribute_name,
                                                          const SessionState& subgraph_session_state) = 0;

  virtual void Sampling__Init(contrib::transformers::Sampling* p, const OpKernelInfo& info) = 0;
  virtual Status Sampling__Compute(const contrib::transformers::Sampling* p, OpKernelContext* ctx) = 0;
  virtual Status Sampling__SetupSubgraphExecutionInfo(contrib::transformers::Sampling* p, const SessionState& session_state, const std::string& attribute_name, const SessionState& subgraph_session_state) = 0;
#endif  // !defined(DISABLE_GENERATION_OPS)

#ifdef ENABLE_ATEN
  virtual Status ATen__Compute(const contrib::ATen* p, OpKernelContext* p_ctx) = 0;
#endif
#endif

#ifdef ENABLE_TRAINING_OPS
  virtual Status contrib__Group__Compute(const contrib::Group* p, OpKernelContext* context) = 0;
  virtual Status contrib__PassThrough__Compute(const contrib::PassThrough* p, OpKernelContext* context) = 0;
  virtual void contrib__VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) = 0;
  virtual void contrib__GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) = 0;
  virtual void contrib__GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, TensorShapeVector& new_shape, std::vector<size_t>& permutations) = 0;
  virtual Status contrib__PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) = 0;
  // From cpu/optimizer/adamwbase.h
  virtual Status contrib__AdamWOptimizerBase__PrepareForCompute(const contrib::AdamWOptimizerBase* p, OpKernelContext* ctx, contrib__AdamWOptimizerBase__Prepare& prepare) = 0;
  // From cpu/optimizer/sgdbase.h
  virtual Status contrib__SGDOptimizerV2Base__PrepareForCompute(const contrib::SGDOptimizerV2Base* p, OpKernelContext* ctx, contrib__SGDOptimizerV2Base__Prepare& prepare) = 0;

  // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
  // 2). this is needed by inference for other purpose.
  virtual void contrib__ShrunkenGatherCommon__CheckInput(const contrib::ShrunkenGatherCommon* p,
                                                         const Tensor* input_tensor, const Tensor* indices_tensor,
                                                         int64_t axis_in) const = 0;
#endif

#ifdef ENABLE_TRAINING
  virtual void contrib__record_event_in_tensor(const Tensor& event_id_tensor) = 0;
  virtual void contrib__wait_event_in_tensor(const Tensor& event_id_tensor) = 0;
  virtual Status contrib__YieldOp__Compute(const contrib::YieldOp* p, OpKernelContext* context) = 0;

  // From aten_op.h
  virtual bool contrib__IsATenOperatorExecutorInitialized() = 0;
  virtual Status contrib__ExecuteReduceSumATen(OpKernelContext* p_ctx, const gsl::span<const int64_t>& axes, bool keepdims) = 0;
#endif

#ifdef ENABLE_TRITON
  virtual Status contrib__TritonOp__Compute(const contrib::TritonOp* p, OpKernelContext* context) = 0;
  virtual bool contrib__IsTritonOpExecutorInitialized() = 0;
  virtual Status contrib__ExecuteTritonOpByFuncName(
      OpKernelContext* p_ctx, const std::string& func_name, size_t input_count, size_t output_count,
      const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs) = 0;
#endif

#endif
};

#ifdef SHARED_PROVIDER

extern ProviderHostCPU& g_host_cpu;

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
namespace GatherElements {
inline Status ValidateInputShapes(const TensorShape& input_data_shape,
                                  const TensorShape& indices_shape,
                                  int64_t axis) { return g_host_cpu.GatherElements__ValidateInputShapes(input_data_shape, indices_shape, axis); }
}  // namespace GatherElements

namespace cumsum_op {
inline Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) { return g_host_cpu.cumsum_op__GetAxis(axis_tensor, input_rank, axis_out); }
}  // namespace cumsum_op

inline Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) { return g_host_cpu.CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr); }

// From onehot.h
inline Status ValidateInputs(const Tensor* depth, const Tensor* values) { return g_host_cpu.ValidateInputs(depth, values); }
inline Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                                 int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                                 TensorShapeVector& output_shape) { return g_host_cpu.PrepareOutputShape(indices, depth_val, axis, prefix_dim_size, suffix_dim_size, output_shape); }

#ifdef ENABLE_TRAINING_OPS
namespace contrib {
inline void VerifyLogitWeightAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, const TensorShape* weight_shape) { g_host_cpu.contrib__VerifyLogitWeightAndLabelShape(logit_shape, label_shape, weight_shape); }
inline void GetNDCFromLogitAndLabelShape(const TensorShape& logit_shape, const TensorShape& label_shape, int64_t& N_D, int64_t& C) { g_host_cpu.contrib__GetNDCFromLogitAndLabelShape(logit_shape, label_shape, N_D, C); }
inline void GetPermutationAndShape(bool ncd_to_ndc, const TensorShape& tensor_shape, TensorShapeVector& new_shape, std::vector<size_t>& permutations) { g_host_cpu.contrib__GetPermutationAndShape(ncd_to_ndc, tensor_shape, new_shape, permutations); }
inline Status PrepareForTrainingCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims, int& after_dims_including_split_axis, int& after_dims_excluding_split, std::vector<int64_t>& split_sizes) { return g_host_cpu.contrib__PrepareForTrainingCompute(input_shape, num_outputs, axis, before_dims, after_dims_including_split_axis, after_dims_excluding_split, split_sizes); }
}  // namespace contrib
#endif

#ifdef ENABLE_TRAINING
namespace contrib {
inline void record_event_in_tensor(const Tensor& event_id_tensor) { return g_host_cpu.contrib__record_event_in_tensor(event_id_tensor); }
inline void wait_event_in_tensor(const Tensor& event_id_tensor) { return g_host_cpu.contrib__wait_event_in_tensor(event_id_tensor); }

// From aten_op.h
inline bool IsATenOperatorExecutorInitialized() { return g_host_cpu.contrib__IsATenOperatorExecutorInitialized(); }
inline Status ExecuteReduceSumATen(OpKernelContext* p_ctx, const gsl::span<const int64_t>& axes, bool keepdims) {
  return g_host_cpu.contrib__ExecuteReduceSumATen(p_ctx, axes, keepdims);
}
}  // namespace contrib
#endif  // ENABLE_TRAINING

#ifdef ENABLE_TRITON
namespace contrib {
inline bool IsTritonOpExecutorInitialized() { return g_host_cpu.contrib__IsTritonOpExecutorInitialized(); }
inline Status ExecuteTritonOpByFuncName(OpKernelContext* p_ctx, const std::string& func_name, size_t input_count,
                                        size_t output_count,
                                        const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs) {
  return g_host_cpu.contrib__ExecuteTritonOpByFuncName(p_ctx, func_name, input_count, output_count, kwargs);
}
}  // namespace contrib
#endif  // ENABLE_TRITON

#endif  // USE_CUDA || USE_CUDA_PROVIDER_INTERFACE
#endif

}  // namespace onnxruntime
