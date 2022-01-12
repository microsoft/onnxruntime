#include "core/providers/cpu/rnn/lstm_base.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/providers/cpu/rnn/uni_directional_lstm.h"

namespace onnxruntime {
namespace contrib {

using namespace rnn::detail;

class DynamicQuantizeLSTM : public OpKernel, public LSTMBase {
 public:
  DynamicQuantizeLSTM(const OpKernelInfo& info) : OpKernel(info), LSTMBase(info) {}

  Status PrePack(const Tensor& tensor, int input_idx,
                 AllocatorPtr alloc, /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  Status Compute(OpKernelContext* context) const override;

  ~DynamicQuantizeLSTM() override = default;

 private:
  Status TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed,
                        bool& is_weight_signed, AllocatorPtr& alloc);

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  PackedWeights packed_W_;
  PackedWeights packed_R_;
  bool is_W_signed_;
  bool is_R_signed_;
};

Status DynamicQuantizeLSTM::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights,
                                           bool& is_packed, bool& is_weight_signed, AllocatorPtr& alloc) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, input_size, 4*hidden_size]
  // recurrence weights: [num_directions, hidden_size, 4*hidden_size]
  const size_t K = static_cast<size_t>(shape[1]);
  const size_t N = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_) * 4)) {
    return Status::OK();
  }

  is_weight_signed = weights.IsDataType<int8_t>();
  const size_t packed_weights_size = MlasGemmPackBSize(N, K, false /*AIsSigned*/, is_weight_signed);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  size_t packed_weights_data_size = SafeInt<size_t>(packed_weights_size) * num_directions_;
  auto* packed_weights_data = alloc->Alloc(packed_weights_data_size);

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_weights_data, 0, packed_weights_data_size);

  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.buffer_size_ = packed_weights_data_size;
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = static_cast<const uint8_t*>(weights.DataRaw());
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(N, K, weights_data, N, false /*AIsSigned*/, is_weight_signed, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

static void UseSharedPrePackedBuffersImpl(std::vector<BufferUniquePtr>& prepacked_buffers,
                                          rnn::detail::PackedWeights& packed_tensor) {
  packed_tensor.buffer_ = std::move(prepacked_buffers[0]);
}

Status DynamicQuantizeLSTM::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                    /*out*/ bool& is_packed,
                                    /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  if (input_idx == 1) {
    ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_W_, is_packed, is_W_signed_, alloc));

    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (is_packed && share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_W_.buffer_));
      prepacked_weights->buffer_sizes_.push_back(packed_W_.buffer_size_);
    }
  } else if (input_idx == 2) {
    ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_R_, is_packed, is_R_signed_, alloc));

    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (is_packed && share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_R_.buffer_));
      prepacked_weights->buffer_sizes_.push_back(packed_R_.buffer_size_);
    }
  }

  return Status::OK();
}

Status DynamicQuantizeLSTM::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                      int input_idx,
                                                      /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    UseSharedPrePackedBuffersImpl(prepacked_buffers, packed_W_);
  } else if (input_idx == 2) {
    used_shared_buffers = true;
    UseSharedPrePackedBuffersImpl(prepacked_buffers, packed_R_);
  }

  return Status::OK();
}

#define WeightCheck(weight_shape, weight_name)                                                                                              \
  if ((weight_shape.NumDimensions() != 1 && weight_shape.NumDimensions() != 2) ||                                                           \
      (weight_shape.NumDimensions() == 2 && weight_shape[1] != static_cast<int64_t>(hidden_size_) * 4) ||                                                         \
      weight_shape[0] != num_directions_) {                                                                                                 \
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,                                                                                   \
                           "Input ", #weight_name, " must have shape {", num_directions_, "} for per-tensor/layer quantization or shape {", \
                           num_directions_, ", 4*", hidden_size_, "} for per-channel quantization. Actual:", weight_shape);                 \
  }

#define ZeroPointCheck(w_zp, zp_shape, is_W_signed, weight_name)                                                                           \
  if (zp_shape.NumDimensions() == 2) {                                                                                                     \
    const int64_t zp_size = zp_shape.Size();                                                                                               \
    const uint8_t* w_zp_data = static_cast<const uint8_t*>(w_zp->DataRaw());                                                               \
    if (is_W_signed) {                                                                                                                     \
      for (int64_t i = 0; i < zp_size; i++) {                                                                                              \
        if (w_zp_data[i] != 0) {                                                                                                           \
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "DynamicQuantizeLSTM : ", #weight_name, "Weight zero point must be zero"); \
        }                                                                                                                                  \
      }                                                                                                                                    \
    } else {                                                                                                                               \
      const uint8_t W_zero_point_value = w_zp_data[0];                                                                                     \
      for (int64_t i = 1; i < zp_size; i++) {                                                                                              \
        if (w_zp_data[i] != W_zero_point_value) {                                                                                          \
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "DynamicQuantizeLSTM : ", #weight_name, "Weight point must be constant");  \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
  }

Status DynamicQuantizeLSTM::Compute(OpKernelContext* context) const {
  // weights. [num_directions, input_size, 4*hidden_size]
  const Tensor* W = packed_W_.buffer_ ? nullptr : context->Input<Tensor>(1);
  // recurrence weights. [num_directionshidden_size, 4*hidden_size]
  const Tensor* R = packed_R_.buffer_ ? nullptr : context->Input<Tensor>(2);

  const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
  const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

  const Tensor* w_scale = context->Input<Tensor>(8);
  const Tensor* w_zp = context->Input<Tensor>(9);
  const Tensor* r_scale = context->Input<Tensor>(10);
  const Tensor* r_zp = context->Input<Tensor>(11);

  const TensorShape& W_zp_shape = w_zp->Shape();
  const TensorShape& R_zp_shape = w_zp->Shape();
  const TensorShape& W_scale_shape = w_scale->Shape();
  const TensorShape& R_scale_shape = r_scale->Shape();

  WeightCheck(W_zp_shape, W_zero_point);
  WeightCheck(R_zp_shape, R_zero_point);
  WeightCheck(W_scale_shape, W_scale);
  WeightCheck(W_scale_shape, R_scale);

  const bool is_W_signed = (W != nullptr) ? W->IsDataType<int8_t>() : is_W_signed_;
  const bool is_R_signed = (R != nullptr) ? R->IsDataType<int8_t>() : is_R_signed_;

  ZeroPointCheck(w_zp, W_zp_shape, is_W_signed, Input);
  ZeroPointCheck(r_zp, R_zp_shape, is_R_signed, Recurrent);

  size_t W_scale_size = W_scale_shape.NumDimensions() == 2 ? W_scale_shape[1] : 1;
  size_t R_scale_size = R_scale_shape.NumDimensions() == 2 ? R_scale_shape[1] : 1;

  QuantizationParameter quant_para_W_1(w_scale->Data<float>(),
                                       static_cast<const uint8_t*>(w_zp->DataRaw()),
                                       is_W_signed,
                                       W_scale_size);
  QuantizationParameter quant_para_R_1(r_scale->Data<float>(),
                                       static_cast<const uint8_t*>(r_zp->DataRaw()),
                                       is_R_signed,
                                       R_scale_size);

  const uint8_t* W_data = W != nullptr ? static_cast<const uint8_t*>(W->DataRaw()) : nullptr;
  const uint8_t* R_data = R != nullptr ? static_cast<const uint8_t*>(R->DataRaw()) : nullptr;

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
    quant_para_W_2.scale += W_scale_size;
    quant_para_R_2.scale += R_scale_size;

    quant_para_W_2.zero_point += W_scale_size;  // zero_point and scale have same size
    quant_para_R_2.zero_point += R_scale_size;  // zero_point and scale have same size

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
