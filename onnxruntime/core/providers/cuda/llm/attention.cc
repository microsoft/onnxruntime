// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_naive.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;
1Code has alerts.Press enter to view.using namespace ::onnxruntime::common;
1Code has alerts.Press enter to view.using namespace ONNX_NAMESPACE;
1Code has alerts.Press enter to view.using namespace onnxruntime::attention_helper;
1Code has alerts.Press enter to view.

    namespace onnxruntime {
  namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      23,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

  REGISTER_KERNEL_TYPED(float)
  REGISTER_KERNEL_TYPED(MLFloat16)
  REGISTER_KERNEL_TYPED(BFloat16)

  template <typename T>
  Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
    is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
    // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
    // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
    kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
    q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
    int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
    qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                                 ? static_cast<QKMatMulOutputMode>(mode)
                                 : QKMatMulOutputMode::kNone;
    ORT_ENFORCE(qk_matmul_output_mode_ == QKMatMulOutputMode::kNone ||
                    qk_matmul_output_mode_ == QKMatMulOutputMode::kQK ||
                    qk_matmul_output_mode_ == QKMatMulOutputMode::kQKMask ||
                    qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftCap ||
                    qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftMax,
                "qk_matmul_output_mode must be 0, 1, 2, or 3.");
    // The default scale depends on the input dimensions. It is set to nan to indicate that it should be computed.
    scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
    1Code has alerts.Press enter to view.softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
    softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
    ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
  }

  template <typename T>
  Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
    const Tensor* Q = context->Input<Tensor>(0);
    const Tensor* K = context->Input<Tensor>(1);
    const Tensor* V = context->Input<Tensor>(2);
    const Tensor* attn_mask = context->Input<Tensor>(3);
    const Tensor* past_key = context->Input<Tensor>(4);
    const Tensor* past_value = context->Input<Tensor>(5);

    AttentionParameters parameters;
    TensorShape y_shape;
    TensorShape present_key_shape;
    TensorShape present_value_shape;
    TensorShape output_qk_shape;

    ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                    Q,
                    K,
                    V,
                    attn_mask,
                    past_key,
                    past_value,
                    is_causal_,
                    softcap_,
                    softmax_precision_,
                    qk_matmul_output_mode_,
                    kv_num_heads_,
                    q_num_heads_,
                    scale_,
                    parameters,
                    y_shape,
                    present_key_shape,
                    present_value_shape,
                    output_qk_shape)
                    .IsOK(),
                "Output shapes for Attention could not be computed.");

    Tensor* Y = context->Output(0, y_shape);
    Tensor* present_key = context->Output(1, present_key_shape);
    Tensor* present_value = context->Output(2, present_value_shape);
    Tensor* output_qk = parameters.qk_matmul_output_mode == QKMatMulOutputMode::kNone
                            ? nullptr
                            : context->Output(3, output_qk_shape);

    // To reuse the existing attention-cuda implementation in contrib ops,
    // map the parameters to contribop_parameters.
    onnxruntime::contrib::AttentionParameters contribop_parameters;
    contribop_parameters.batch_size = parameters.batch_size;
    contribop_parameters.sequence_length = parameters.q_sequence_length;
    contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
    contribop_parameters.past_sequence_length = parameters.past_sequence_length;
    contribop_parameters.total_sequence_length = parameters.total_sequence_length;
    // max_sequence_length: For non-buffer-sharing case, this equals total_sequence_length (the present KV cache size)
    contribop_parameters.max_sequence_length = parameters.total_sequence_length;
    contribop_parameters.input_hidden_size = 0;  // Not applicable - new Attention op takes pre-projected Q/K/V
    contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
    contribop_parameters.head_size = parameters.head_size;
    contribop_parameters.v_head_size = parameters.v_head_size;
    contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
    contribop_parameters.num_heads = parameters.q_num_heads;
    contribop_parameters.rotary_dim = 0;
    contribop_parameters.num_splits = 1;
    contribop_parameters.beam_width = 1;
    contribop_parameters.is_unidirectional = parameters.is_causal;
    contribop_parameters.past_present_share_buffer = false;  // New Attention op doesn't share buffer
    contribop_parameters.is_packed_qkv = false;
    contribop_parameters.do_rotary = false;

    // Determine mask type from attn_mask input
    if (attn_mask == nullptr) {
      contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
    } else {
      const auto& mask_dims = attn_mask->Shape().GetDims();
      if (mask_dims.size() == 2 && mask_dims[0] == parameters.batch_size &&
          mask_dims[1] == parameters.total_sequence_length) {
        contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_2D_KEY_PADDING;
      } else {
        contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_UNKNOWN;
      }
    }

    // Determine broadcast flags for attention_bias (if it exists)
    // Note: The new Attention op uses attn_mask, not attention_bias
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = false;

    contribop_parameters.mask_filter_value = -10000.0f;
    contribop_parameters.scale = parameters.scale;
    contribop_parameters.use_tf32 = UseTF32();

    // QKV format: Always Q_K_V_BSNH for separate Q, K, V inputs
    // (3D inputs get internally transposed to 4D BNSH, then treated as BSNH)
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;

    // TODO(titaiwang): Continue on these parameters
    // Construct AttentionData to pass to QkvToContext
    typedef typename ToCudaType<T>::MappedType CudaT;
    onnxruntime::contrib::cuda::AttentionData<CudaT> data;

    // Set input pointers
    data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
    data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
    data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
    data.mask_index = (attn_mask == nullptr) ? nullptr : attn_mask->Data<int>();
    data.mask_index_dims = (attn_mask == nullptr) ? gsl::span<const int64_t>() : attn_mask->Shape().GetDims();
    data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());

    // Set output pointers
    data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());

    // Set additional fields
    data.bias = nullptr;            // New Attention op doesn't have bias
    data.attention_bias = nullptr;  // New Attention op uses attn_mask, not attention_bias
    data.qkv_format = contribop_parameters.qkv_format;

    // TODO: Determine which kernel to use (Flash Attention, Memory Efficient Attention, etc.)
    // For now, set flags to false and let QkvToContext use the unfused path
    data.use_flash_attention = false;
    data.use_memory_efficient_attention = false;
    data.fused_runner = nullptr;
    data.fused_cross_attention_kernel = nullptr;

    // Call QkvToContext to perform the attention computation
    auto& device_prop = GetDeviceProp();
    cublasHandle_t cublas = GetCublasHandle(context);
    cudnnHandle_t cudnn = GetCudnnHandle(context);

    // QkvToContext takes two template parameters: T for computation type, QK for output_qk type
    // For now, both are the same type (CudaT)
    return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
        device_prop, cublas, cudnn, context->GetComputeStream(), contribop_parameters, data);
  }
  }  // namespace cuda
}  // namespace onnxruntime
