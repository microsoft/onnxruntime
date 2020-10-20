#include "core/providers/cpu/rnn/lstm_base.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/providers/cpu/rnn/uni_directional_lstm.h"

namespace onnxruntime {
namespace contrib {

using namespace rnn::detail;

class DynamicQuantizeLSTM : public OpKernel, public LSTMBase {
 public:
  DynamicQuantizeLSTM(const OpKernelInfo& info) : OpKernel(info), LSTMBase(info) {}

#if !defined(USE_MKLML_FOR_BLAS)
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

#if !defined(USE_MKLML_FOR_BLAS)
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

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

Status DynamicQuantizeLSTM::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  auto& logger = context->Logger();

  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor* W = packed_W_.buffer_ ? nullptr : context->Input<Tensor>(1);
  // weights. [num_directions, 4*hidden_size, input_size]
  const Tensor* R = packed_R_.buffer_ ? nullptr : context->Input<Tensor>(2);
  // recurrence weights. [num_directions, 4*hidden_size, hidden_size]

  // optional
  const Tensor* B = context->Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  const Tensor* sequence_lens = context->Input<Tensor>(4);  // [batch_size]
  const Tensor* initial_h = context->Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]
  const Tensor* initial_c = context->Input<Tensor>(6);      // initial cell. [num_directions, batch_size, hidden_size]
  const Tensor* P = context->Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]
  const Tensor* w_scale = context->Input<Tensor>(8);
  const Tensor* w_zp = context->Input<Tensor>(9);
  const Tensor* r_scale = context->Input<Tensor>(10);
  const Tensor* r_zp = context->Input<Tensor>(11);

  const auto& X_shape = X.Shape();

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

  const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
  const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

  Status status = ValidateInputs(X, W_shape, R_shape, B, sequence_lens, initial_h, initial_c, P, batch_size);
  ORT_RETURN_IF_ERROR(status);

  // LSTM outputs are optional but must be in the same order
  TensorShape Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = context->Output(/*index*/ 0, Y_dims);

  TensorShape Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = context->Output(/*index*/ 1, Y_h_dims);

  TensorShape Y_c_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_c = context->Output(/*index*/ 2, Y_c_dims);

  // Reset output and return if max sequence length is 0
  if (sequence_lens != nullptr) {
    int32_t max_sequence_length = *std::max_element(sequence_lens->Data<int32_t>(),
                                                    sequence_lens->Data<int32_t>() + sequence_lens->Shape().Size());
    if (max_sequence_length == 0) {
      if (Y != nullptr)
        std::fill_n(Y->MutableData<float>(), Y_dims.Size(), 0.0f);
      if (Y_h != nullptr)
        std::fill_n(Y_h->MutableData<float>(), Y_h_dims.Size(), 0.0f);
      if (Y_c != nullptr)
        std::fill_n(Y_c->MutableData<float>(), Y_c_dims.Size(), 0.0f);
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  status = context->GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  const uint8_t* input_weights = nullptr;
  const uint8_t* recurrent_weights = nullptr;
  QuantizationParameter quant_para_W{w_scale->Data<float>(), w_zp ? static_cast<const uint8_t*>(w_zp->DataRaw()) : nullptr, weights_signed_};
  QuantizationParameter quant_para_R{r_scale->Data<float>(), r_zp ? static_cast<const uint8_t*>(r_zp->DataRaw()) : nullptr, weights_signed_};
  if (W != nullptr) {
    input_weights = static_cast<const uint8_t*>(W->DataRaw());
    quant_para_W.b_is_signed = W->IsDataType<int8_t>();
  }

  if (R != nullptr) {
    recurrent_weights = static_cast<const uint8_t*>(R->DataRaw());
    quant_para_R.b_is_signed = R->IsDataType<int8_t>();
  }

  gsl::span<const float> bias = B != nullptr ? B->DataAsSpan<float>() : gsl::span<const float>();
  gsl::span<const float> peephole_weights = P != nullptr ? P->DataAsSpan<float>() : gsl::span<const float>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 4 * hidden_size_ * input_size;
  const size_t hidden_weights_size_per_direction = 4 * hidden_size_ * hidden_size_;
  const size_t bias_size_per_direction = 8 * hidden_size_;
  const size_t peephole_weights_size_per_direction = 3 * hidden_size_;

  GemmWeights<uint8_t> input_weights_1(0, input_weights, input_weights_size_per_direction, packed_W_, &quant_para_W);
  GemmWeights<uint8_t> recurrent_weights_1(0, recurrent_weights, hidden_weights_size_per_direction, packed_R_, &quant_para_R);

  gsl::span<const float> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);
  gsl::span<const float> peephole_weights_1 =
      peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(0, peephole_weights_size_per_direction);

  gsl::span<const float> input = X.DataAsSpan<float>();
  gsl::span<const int> sequence_lens_span =
      sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>() : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const float> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<float>() : gsl::span<const float>();
  gsl::span<const float> initial_hidden_1 =
      initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  const size_t initial_cell_size_per_direction = batch_size * hidden_size_;
  gsl::span<const float> initial_cell = initial_c != nullptr ? initial_c->DataAsSpan<float>() : gsl::span<const float>();
  gsl::span<const float> initial_cell_1 =
      initial_cell.empty() ? initial_cell : initial_cell.subspan(0, initial_cell_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<float> output = Y != nullptr ? Y->MutableDataAsSpan<float>() : gsl::span<float>();
  gsl::span<float> output_1 =
      output.empty() ? output : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalLstm needs somewhere to write output, so even if we aren't returning Y_h and Y_c
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<float> local_hidden_output;
  gsl::span<float> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<float>()
          : Allocate(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<float> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  const size_t last_cell_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<float> local_last_cell;
  gsl::span<float> last_cell = Y_c ? Y_c->MutableDataAsSpan<float>() : Allocate(alloc, last_cell_size_per_direction * num_directions_, local_last_cell);

  gsl::span<float> last_cell_1 = last_cell.subspan(0, last_cell_size_per_direction);

  if (direction_ == Direction::kBidirectional) {
    GemmWeights<uint8_t> input_weights_2(1, input_weights, input_weights_size_per_direction, packed_W_, &quant_para_W);
    GemmWeights<uint8_t> recurrent_weights_2(1, recurrent_weights, hidden_weights_size_per_direction, packed_R_, &quant_para_R);

    // spans for second direction
    gsl::span<const float> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);
    gsl::span<const float> peephole_weights_2 =
        peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(peephole_weights_size_per_direction, peephole_weights_size_per_direction);

    gsl::span<const float> initial_hidden_2 =
        initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(initial_hidden_size_per_direction, initial_hidden_size_per_direction);
    gsl::span<const float> initial_cell_2 =
        initial_cell.empty() ? initial_cell : initial_cell.subspan(initial_cell_size_per_direction, initial_cell_size_per_direction);
    gsl::span<float> output_2 =
        output.empty() ? output : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<float> hidden_output_2 =
        hidden_output.subspan(hidden_output_size_per_direction, hidden_output_size_per_direction);
    gsl::span<float> last_cell_2 = last_cell.subspan(last_cell_size_per_direction, last_cell_size_per_direction);

    lstm::UniDirectionalLstm<float> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                       Direction::kForward, input_forget_, bias_1, peephole_weights_1, initial_hidden_1,
                                       initial_cell_1, activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                       activation_funcs_.Entries()[2], clip_, thread_pool);

    lstm::UniDirectionalLstm<float> bw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                       Direction::kReverse, input_forget_, bias_2, peephole_weights_2, initial_hidden_2,
                                       initial_cell_2, activation_funcs_.Entries()[3], activation_funcs_.Entries()[4],
                                       activation_funcs_.Entries()[5], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1,
               hidden_output_1, last_cell_1);
    bw.Compute(input, sequence_lens_span, num_directions_, input_weights_2, recurrent_weights_2, output_2,
               hidden_output_2, last_cell_2);
  } else {
    lstm::UniDirectionalLstm<float> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_, direction_,
                                       input_forget_, bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
                                       activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                       activation_funcs_.Entries()[2], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1,
               hidden_output_1, last_cell_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  // these always get written to regardless of whether we're returning them as optional output or not
  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);
  DumpMatrix("Y_c", last_cell.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
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
