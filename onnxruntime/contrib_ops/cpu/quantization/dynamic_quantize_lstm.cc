#include "core/providers/cpu/rnn/lstm_base.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/providers/cpu/rnn/uni_directional_lstm.h"

namespace onnxruntime {
namespace contrib {

using namespace rnn::detail;

class DynamicQuantizeLSTM : public OpKernel, public LSTMBase {
 public:
  DynamicQuantizeLSTM(const OpKernelInfo& info) : OpKernel(info), LSTMBase(info) {}

#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;
#endif

  Status Compute(OpKernelContext* context) const override;

  ~DynamicQuantizeLSTM() override = default;

 private:
  Status TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed);

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  PackedWeights packed_W_;
  PackedWeights packed_R_;
  bool weights_signed_;
};

Status DynamicQuantizeLSTM::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, input_size, 4*hidden_size]
  // recurrence weights: [num_directions, hidden_size, 4*hidden_size]
  const size_t K = static_cast<size_t>(shape[1]);
  const size_t N = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_ * 4))) {
    return Status::OK();
  }

  weights_signed_ = weights.IsDataType<int8_t>();
  const size_t packed_weights_size = MlasGemmPackBSize(N, K, weights_signed_);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = alloc->Alloc(SafeInt<size_t>(packed_weights_size) * num_directions_);
  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = static_cast<const uint8_t*>(weights.DataRaw());
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(N, K, weights_data, N, weights_signed_, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
Status DynamicQuantizeLSTM::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  if (input_idx == 1) {
    return TryPackWeights(tensor, packed_W_, is_packed);
  } else if (input_idx == 2) {
    return TryPackWeights(tensor, packed_R_, is_packed);
  }

  return Status::OK();
}
#endif

Status DynamicQuantizeLSTM::Compute(OpKernelContext* context) const {
  const Tensor* W = packed_W_.buffer_ ? nullptr : context->Input<Tensor>(1);
  // weights. [num_directions, input_size, 4*hidden_size]
  const Tensor* R = packed_R_.buffer_ ? nullptr : context->Input<Tensor>(2);
  // recurrence weights. [num_directionshidden_size, 4*hidden_size, ]

  const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
  const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

  const Tensor* w_scale = context->Input<Tensor>(8);
  const Tensor* w_zp = context->Input<Tensor>(9);
  const Tensor* r_scale = context->Input<Tensor>(10);
  const Tensor* r_zp = context->Input<Tensor>(11);

  const auto& W_scale_shape = w_scale->Shape();
  const auto& R_scale_shape = r_scale->Shape();

#if 0  // TODO: Enable Per-Column after MLAS kernel change is done
  int64_t per_column_size = num_directions_ * 4 * hidden_size_;
  QuantizationType quant_W_type = W_scale_shape.Size() == per_column_size ? QuantizationType::PerColumn : QuantizationType::PerTensor;
  QuantizationType quant_R_type = R_scale_shape.Size() == per_column_size ? QuantizationType::PerColumn : QuantizationType::PerTensor;
#else
  QuantizationType quant_W_type = QuantizationType::PerTensor;
  QuantizationType quant_R_type = QuantizationType::PerTensor;
#endif
  QuantizationParameter quant_para_W_1(w_scale->Data<float>(),
                                       static_cast<const uint8_t*>(w_zp->DataRaw()),
                                       weights_signed_,
                                       quant_W_type);
  QuantizationParameter quant_para_R_1(r_scale->Data<float>(),
                                       r_zp ? static_cast<const uint8_t*>(r_zp->DataRaw()) : nullptr,
                                       weights_signed_,
                                       quant_R_type);

  const uint8_t* W_data = nullptr;
  const uint8_t* R_data = nullptr;
  if (W != nullptr) {
    W_data = static_cast<const uint8_t*>(W->DataRaw());
    quant_para_W_1.is_signed = W->IsDataType<int8_t>();
  }

  if (R != nullptr) {
    R_data = static_cast<const uint8_t*>(R->DataRaw());
    quant_para_R_1.is_signed = R->IsDataType<int8_t>();
  }

  // spans for first direction
  const size_t W_size_per_direction = W_shape[1] * W_shape[2];
  const size_t R_size_per_direction = R_shape[1] * R_shape[2];

  GemmWeights<uint8_t> W_1(0, W_data, W_size_per_direction, packed_W_, &quant_para_W_1);
  GemmWeights<uint8_t> R_1(0, R_data, R_size_per_direction, packed_R_, &quant_para_R_1);

  GemmWeights<uint8_t> W_2;
  GemmWeights<uint8_t> R_2;

  QuantizationParameter quant_para_W_2(quant_para_W_1);
  QuantizationParameter quant_para_R_2(quant_para_R_1);

  if (direction_ == Direction::kBidirectional) {
    quant_para_W_2.scale += W_scale_shape.SizeFromDimension(1);
    quant_para_R_2.scale += R_scale_shape.SizeFromDimension(1);

    quant_para_W_2.zero_point += W_scale_shape.SizeFromDimension(1);
    quant_para_R_2.zero_point += R_scale_shape.SizeFromDimension(1);

    W_2.Init(1, W_data, W_size_per_direction, packed_W_, &quant_para_W_2);
    R_2.Init(1, R_data, R_size_per_direction, packed_R_, &quant_para_R_2);
  }

  return LSTMBase::ComputeImpl<float, uint8_t>(*context, W_1, W_2, R_1, R_2);
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeLSTM,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeLSTM);

}  // namespace contrib
}  // namespace onnxruntime
