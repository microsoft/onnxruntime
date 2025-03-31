#include "attention_utils.h"
#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/cpu/math/element_wise_ops.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// Reshape Q/K/V from BxSxD to BxSxNxH
inline Status Reshape_BSD_to_BSNH(Tensor* qkv,
                                  int batch_size,
                                  int sequence_length,
                                  int num_heads,
                                  int head_size) {
  qkv->Reshape(TensorShape({batch_size, sequence_length, num_heads, head_size}));
  return Status::OK();
}

// Transpose Q/K/V from BxSxNxH to BxNxSxH
inline Status Transpose_BSNH_to_BNSH(const Tensor* qkv,
                                     OrtValue& qkv_transposed,
                                     concurrency::ThreadPool* tp) {
  std::vector<size_t> permutations({0, 2, 1, 3});
  gsl::span<const size_t> permutations_span{permutations};
  size_t from = 2, to = 1;
  SingleAxisTranspose(permutations, *qkv, *qkv_transposed.GetMutable<Tensor>(), from, to, nullptr, tp);
  return Status::OK();
}

// Add bias + transpose for each of Q/K/V
template <typename T>
Status AddBiasTranspose(const Tensor* qkv,                   // Input: Q/K/V data - query is BxSxD, key is BxLxD, value is BxLxD_v
                        const T* qkv_bias,                   // Input: QKV bias - bias is (D + D + D_v)
                        OrtValue& qkv_with_bias_transposed,  // Output: Q/K/V data - query is BxNxSxH, key is BxNxLxH, value is BxNxLxH_v
                        int bias_offset,                     // bias offset to enter qkv_bias
                        int batch_size,                      // batch size
                        int sequence_length,                 // sequence_length for Q, kv_sequence_length for K/V
                        int num_heads,                       // num heads
                        int head_size,                       // head_size for Q/K, v_head_size for V
                        int hidden_size,                     // hidden_size for Q/K, v_hidden_size for V
                        OpKernelContext* context) {
  // Note: the comments below will refer to Q's dimensions for simplicity
  auto element_type = DataTypeImpl::GetType<T>();
  constexpr size_t element_size = sizeof(T);
  ProcessBroadcastSpanFuncs add_funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.ScalarInput0<float>() + per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().array() + per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>() + per_iter_bh.EigenInput1<float>();
      }};  // For element-wise add

  // Allocate space for output of Q(BS, D) + bias(D)
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  std::vector<int64_t> old_dims({batch_size, sequence_length, hidden_size});
  gsl::span<const int64_t> old_dims_span{old_dims};
  TensorShape qkv_with_bias_shape(old_dims_span);
  OrtValue qkv_with_bias;
  Tensor::InitOrtValue(element_type, qkv_with_bias_shape, allocator, qkv_with_bias);

  // Get Q's bias from combined bias
  std::vector<int64_t> bias_dims({hidden_size});
  gsl::span<const int64_t> bias_dims_span{bias_dims};
  TensorShape bias_shape(bias_dims_span);
  OrtValue bias;
  Tensor::InitOrtValue(element_type, bias_shape, allocator, bias);
  memcpy(bias.GetMutable<Tensor>()->MutableData<T>(), qkv_bias + bias_offset, hidden_size * element_size);

  // Compute Q(BS, D) + bias(D) as broadcasted element-wise add
  {
    InputBroadcaster input_broadcaster(*bias.GetMutable<Tensor>(), *qkv);
    const InputBroadcaster& const_input_broadcaster = input_broadcaster;
    Tensor& output_tensor = *qkv_with_bias.GetMutable<Tensor>();

    size_t span_size = input_broadcaster.GetSpanSize();
    size_t output_size = static_cast<ptrdiff_t>(output_tensor.Shape().Size());
    void* user_data = nullptr;

    const int loop_len = static_cast<int>(output_size / span_size);
    double unit_cost = 1.0f;
    const auto cost = TensorOpCost{static_cast<double>(input_broadcaster.Input0ElementSize()) * span_size,
                                   static_cast<double>(output_tensor.DataType()->Size()) * span_size,
                                   unit_cost * span_size};
    auto tp = context->GetOperatorThreadPool();
    ThreadPool::TryParallelFor(tp, loop_len, cost,
                               [span_size, &const_input_broadcaster, &output_tensor, &add_funcs, user_data](std::ptrdiff_t first_span,
                                                                                                            std::ptrdiff_t last_span) {
                                 InputBroadcaster segment_input_broadcaster(const_input_broadcaster);
                                 segment_input_broadcaster.AdvanceBy(first_span * span_size);

                                 OutputBroadcaster segment_output_broadcaster(span_size, output_tensor,
                                                                              first_span * span_size, last_span * span_size);

                                 BroadcastHelper segment_helper(segment_input_broadcaster, segment_output_broadcaster, user_data);
                                 BroadcastLooper(segment_helper, add_funcs);
                               });
  }

  // Reshape Q from BxSxD to BxSxNxH
  ORT_RETURN_IF_ERROR(Reshape_BSD_to_BSNH(qkv_with_bias.GetMutable<Tensor>(), batch_size, sequence_length, num_heads, head_size));

  // Transpose Q from BxSxNxH to BxNxSxH
  auto tp = context->GetOperatorThreadPool();
  ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH(qkv_with_bias.GetMutable<Tensor>(), qkv_with_bias_transposed, tp));

  return Status::OK();
}

// Add bias + reshape for each of Q/K/V
// This is used in decoder_with_past when the sequence length is 1
template <typename T>
Status AddBiasReshape(const Tensor* qkv,        // Input: Q/K/V data - query is BxSxD, key is BxLxD, value is BxLxD_v
                      const T* qkv_bias,        // Input: QKV bias - bias is (D + D + D_v)
                      OrtValue& qkv_with_bias,  // Output: Q/K/V data - query is BxNxSxH, key is BxNxLxH, value is BxNxLxH_v
                      int bias_offset,          // bias offset to enter qkv_bias
                      int batch_size,           // batch size
                      int sequence_length,      // sequence_length for Q, kv_sequence_length for K/V
                      int num_heads,            // num heads
                      int head_size,            // head_size for Q/K, v_head_size for V
                      int hidden_size,          // hidden_size for Q/K, v_hidden_size for V
                      OpKernelContext* context) {
  // Note: the comments below will refer to Q's dimensions for simplicity
  auto element_type = DataTypeImpl::GetType<T>();
  constexpr size_t element_size = sizeof(T);
  ProcessBroadcastSpanFuncs add_funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.ScalarInput0<float>() + per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().array() + per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>() + per_iter_bh.EigenInput1<float>();
      }};  // For element-wise add

  // Get Q's bias from combined bias
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  std::vector<int64_t> bias_dims({hidden_size});
  gsl::span<const int64_t> bias_dims_span{bias_dims};
  TensorShape bias_shape(bias_dims_span);
  OrtValue bias;
  Tensor::InitOrtValue(element_type, bias_shape, allocator, bias);
  auto num_bias_elements = SafeInt<size_t>(hidden_size) * element_size;
  memcpy(bias.GetMutable<Tensor>()->MutableData<T>(), qkv_bias + bias_offset, num_bias_elements);

  // Compute Q(BS, D) + bias(D) as broadcasted element-wise add
  {
    InputBroadcaster input_broadcaster(*bias.GetMutable<Tensor>(), *qkv);
    const InputBroadcaster& const_input_broadcaster = input_broadcaster;
    Tensor& output_tensor = *qkv_with_bias.GetMutable<Tensor>();

    size_t span_size = input_broadcaster.GetSpanSize();
    size_t output_size = static_cast<ptrdiff_t>(output_tensor.Shape().Size());
    void* user_data = nullptr;

    const int loop_len = static_cast<int>(output_size / span_size);
    double unit_cost = 1.0f;
    const auto cost = TensorOpCost{static_cast<double>(input_broadcaster.Input0ElementSize()) * span_size,
                                   static_cast<double>(output_tensor.DataType()->Size()) * span_size,
                                   unit_cost * span_size};
    auto tp = context->GetOperatorThreadPool();
    ThreadPool::TryParallelFor(tp, loop_len, cost,
                               [span_size, &const_input_broadcaster, &output_tensor, &add_funcs, user_data](std::ptrdiff_t first_span,
                                                                                                            std::ptrdiff_t last_span) {
                                 InputBroadcaster segment_input_broadcaster(const_input_broadcaster);
                                 segment_input_broadcaster.AdvanceBy(first_span * span_size);

                                 OutputBroadcaster segment_output_broadcaster(span_size, output_tensor,
                                                                              first_span * span_size, last_span * span_size);

                                 BroadcastHelper segment_helper(segment_input_broadcaster, segment_output_broadcaster, user_data);
                                 BroadcastLooper(segment_helper, add_funcs);
                               });
  }

  // Reshape Q from BxSxD to BxNxSxH
  qkv_with_bias.GetMutable<Tensor>()->Reshape(TensorShape({batch_size, num_heads, sequence_length, head_size}));

  return Status::OK();
}

template <typename T>
Status MaybeTransposeToBNSHAndAddBias(OpKernelContext* context, AllocatorPtr allocator,
                                      int batch_size, int num_heads, int sequence_length, int head_size,
                                      const Tensor* in, const Tensor* bias, int bias_offset, OrtValue& out) {
  auto element_type = DataTypeImpl::GetType<T>();
  std::vector<int64_t> new_dims({batch_size, num_heads, sequence_length, head_size});
  gsl::span<const int64_t> new_dims_span{new_dims};
  TensorShape v_BNLH(new_dims_span);
  Tensor::InitOrtValue(element_type, v_BNLH, allocator, out);
  if (bias == nullptr) {
    std::unique_ptr<Tensor> reshaped;
    if (in->Shape().GetDims().size() == 3) {
      reshaped = std::make_unique<Tensor>(in->DataType(), in->Shape(), const_cast<void*>(in->DataRaw()), in->Location());
      ORT_RETURN_IF_ERROR(Reshape_BSD_to_BSNH(reshaped.get(), batch_size, sequence_length, num_heads, head_size));
    }
    ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH((reshaped == nullptr) ? in : reshaped.get(), out));
  } else {
    const auto* qkv_bias = bias->Data<T>();
    if (sequence_length == 1) {
      ORT_RETURN_IF_ERROR(AddBiasReshape(in, qkv_bias, out, bias_offset, batch_size, sequence_length, num_heads, head_size, num_heads * head_size, context));
    } else {
      ORT_RETURN_IF_ERROR(AddBiasTranspose(in, qkv_bias, out, bias_offset, batch_size, sequence_length, num_heads, head_size, num_heads * head_size, context));
    }
  }
  return Status::OK();
};

template Status MaybeTransposeToBNSHAndAddBias<float>(OpKernelContext* context, AllocatorPtr allocator,
                                                      int batch_size, int num_heads, int sequence_length, int head_size,
                                                      const Tensor* in, const Tensor* bias, int bias_offset, OrtValue& out);

template Status MaybeTransposeToBNSHAndAddBias<MLFloat16>(OpKernelContext* context, AllocatorPtr allocator,
                                                          int batch_size, int num_heads, int sequence_length, int head_size,
                                                          const Tensor* in, const Tensor* bias, int bias_offset, OrtValue& out);

template <typename T>
Status MaybeTransposeToBNSH(AllocatorPtr allocator,
                            int batch_size, int num_heads, int sequence_length, int head_size,
                            const Tensor* in, OrtValue& out) {
  auto element_type = DataTypeImpl::GetType<T>();
  std::vector<int64_t> new_dims({batch_size, num_heads, sequence_length, head_size});
  gsl::span<const int64_t> new_dims_span{new_dims};
  TensorShape v_BNLH(new_dims_span);
  Tensor::InitOrtValue(element_type, v_BNLH, allocator, out);
  std::unique_ptr<Tensor> reshaped;
  if (in->Shape().GetDims().size() == 3) {
    reshaped = std::make_unique<Tensor>(in->DataType(), in->Shape(), const_cast<void*>(in->DataRaw()), in->Location());
    ORT_RETURN_IF_ERROR(Reshape_BSD_to_BSNH(reshaped.get(), batch_size, sequence_length, num_heads, head_size));
  }
  ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH((reshaped == nullptr) ? in : reshaped.get(), out));

  return Status::OK();
};

template Status MaybeTransposeToBNSH<float>(AllocatorPtr allocator,
                                            int batch_size, int num_heads, int sequence_length, int head_size,
                                            const Tensor* in, OrtValue& out);

template Status MaybeTransposeToBNSH<MLFloat16>(AllocatorPtr allocator,
                                                int batch_size, int num_heads, int sequence_length, int head_size,
                                                const Tensor* in, OrtValue& out);

}  // namespace contrib
}  // namespace onnxruntime
