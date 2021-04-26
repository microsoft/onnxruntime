// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "deep_cpu_lstm.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/*
ONNX_OPERATOR_SCHEMA(LSTM)
    .SetDoc(R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor
`i` - input gate
`o` - output gate
`f` - forget gate
`c` - cell gate
`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
`P[iof]`  - P peephole weight vector for input, output, and forget gates
`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)
  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)
  Affine(x)              - alpha*x + beta
  LeakyRelu(x)           - x if x >= 0 else alpha * x
  ThresholdedRelu(x)     - x if x >= alpha else 0
  ScaledTanh(x)          - alpha*Tanh(beta*x)
  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  Softsign(x)            - x/(1 + |x|)
  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)
  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)
  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)
  - Ct = ft (.) Ct-1 + it (.) ct
  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)
  - Ht = ot (.) h(Ct)
)DOC")
    .Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
               "Must be one of forward (default), reverse, or bidirectional.",
               AttributeProto::STRING,
               std::string("forward"))
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL)
    .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
               "for input, output, forget, cell, and hidden. The activation functions must "
               "be one of the activation functions specified above. Optional: See the equations "
               "for default if not specified.",
               AttributeProto::STRINGS,
               OPTIONAL)
    .Attr("activation_alpha",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("activation_beta",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("output_sequence",
               "The sequence output for the hidden is optional if 0. Default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0));
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
               "in the range of [-threshold, +threshold] and is applied to the input "
               "of activations. No clip if not specified.", AttributeProto::FLOAT, OPTIONAL)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0))
    .Input(0, "X",
               "The input sequences packed (and potentially padded) into one 3-D "
               "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    .Input(1, "W",
               "The weight tensor for the gates. Concatenation of `W[iofc]` and "
               "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
               "`[num_directions, 4*hidden_size, input_size]`.", "T")
    .Input(2, "R",
               "The recurrence weight tensor. Concatenation of `R[iofc]` and "
               "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
               "`[num_directions, 4*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
               "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
               "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
               "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
               "specified - assumed to be 0.", "T",
               OpSchema::Optional)
    .Input(4, "sequence_lens",
               "Optional tensor specifying lengths of the sequences in a batch. "
               "If not specified - assumed all sequences in the batch to have "
               "length `seq_length`. It has shape `[batch_size]`.", "T1",
               OpSchema::Optional)
    .Input(5, "initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
                "T", OpSchema::Optional)
    .Input(6, "initial_c",
                "Optional initial value of the cell. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
"T", OpSchema::Optional)
    .Input(7, "P",
                "The weight tensor for peepholes. Concatenation of `P[iof]` and "
                "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
                "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
                "assumed to be 0.", "T",
                OpSchema::Optional)
    .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional);
    .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .Output(2, "Y_c",
                "The last output value of the cell. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to float tensors.")
    .TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");

*/

namespace onnxruntime {

/* LSTM operator */
ONNX_CPU_OPERATOR_KERNEL(LSTM, 7,
                         KernelDefBuilder()
                             .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                   DataTypeImpl::GetTensorType<double>()})
                             .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
                         DeepCpuLstmOp);

using namespace rnn::detail;

// LSTM details

Status DeepCpuLstmOp::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed, AllocatorPtr alloc) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, 4*hidden_size, input_size]
  // recurrence weights: [num_directions, 4*hidden_size, hidden_size]
  const size_t N = static_cast<size_t>(shape[1]);
  const size_t K = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_ * 4))) {
    return Status::OK();
  }

  const size_t packed_weights_size = MlasGemmPackBSize(N, K);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  auto* packed_weights_data = alloc->Alloc(SafeInt<size_t>(packed_weights_size) * num_directions_);
  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = weights.Data<float>();
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(CblasTrans, N, K, weights_data, K, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

static void ParseCachedPrePackedWeights(const PrepackedWeight& cached_prepacked_tensor, rnn::detail::PackedWeights& packed_tensor) {
  packed_tensor.buffer_ = BufferUniquePtr(cached_prepacked_tensor.buffers_[0].get(), BufferDeleter(nullptr));
  packed_tensor.shape_ = cached_prepacked_tensor.shapes_[0];
  packed_tensor.weights_size_ = cached_prepacked_tensor.weights_sizes_[0];
}

Status DeepCpuLstmOp::PrePack(const Tensor& tensor, int input_idx, bool& /*out*/ is_packed,
                              /*out*/ PrepackedWeight& prepacked_weight_for_caching,
                              AllocatorPtr alloc_for_caching) {
  is_packed = false;

  if (tensor.IsDataType<float>()) {
    if (input_idx == 1) {
      bool kernel_owns_prepacked_buffer = (alloc_for_caching == nullptr);
      AllocatorPtr alloc = kernel_owns_prepacked_buffer ? Info().GetAllocator(0, OrtMemTypeDefault) : alloc_for_caching;
      ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_W_, is_packed, alloc));

      if (is_packed && !kernel_owns_prepacked_buffer) {
        prepacked_weight_for_caching.buffers_.push_back(std::move(packed_W_.buffer_));
        prepacked_weight_for_caching.shapes_.push_back(packed_W_.shape_);
        prepacked_weight_for_caching.weights_sizes_.push_back(packed_W_.weights_size_);
        prepacked_weight_for_caching.has_cached_ = true;
        packed_W_.buffer_ = BufferUniquePtr(prepacked_weight_for_caching.buffers_[0].get(), BufferDeleter(nullptr));
      }
    } else if (input_idx == 2) {
      bool kernel_owns_prepacked_buffer = (alloc_for_caching == nullptr);
      AllocatorPtr alloc = kernel_owns_prepacked_buffer ? Info().GetAllocator(0, OrtMemTypeDefault) : alloc_for_caching;
      ORT_RETURN_IF_ERROR(TryPackWeights(tensor, packed_R_, is_packed, alloc));

      if (is_packed && !kernel_owns_prepacked_buffer) {
        prepacked_weight_for_caching.buffers_.push_back(std::move(packed_R_.buffer_));
        prepacked_weight_for_caching.shapes_.push_back(packed_R_.shape_);
        prepacked_weight_for_caching.weights_sizes_.push_back(packed_R_.weights_size_);
        prepacked_weight_for_caching.has_cached_ = true;
        packed_R_.buffer_ = BufferUniquePtr(prepacked_weight_for_caching.buffers_[0].get(), BufferDeleter(nullptr));
      }
    }
  }

  return Status::OK();
}

Status DeepCpuLstmOp::UseCachedPrePackedWeight(const PrepackedWeight& cached_prepacked_weight,
                                               int input_idx,
                                               /*out*/ bool& read_from_cache) {
  read_from_cache = false;

  if (cached_prepacked_weight.has_cached_) {
    if (input_idx == 1) {
      read_from_cache = true;
      ParseCachedPrePackedWeights(cached_prepacked_weight, packed_W_);
    } else if (input_idx == 2) {
      read_from_cache = true;
      ParseCachedPrePackedWeights(cached_prepacked_weight, packed_R_);
    }
  }

  return Status::OK();
}

Status DeepCpuLstmOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  if (X.IsDataType<float>()) {
    const Tensor* W = packed_W_.buffer_ ? nullptr : context->Input<Tensor>(1);
    // weights. [num_directions, 4*hidden_size, input_size]
    const Tensor* R = packed_R_.buffer_ ? nullptr : context->Input<Tensor>(2);
    // recurrence weights. [num_directions, 4*hidden_size, hidden_size]

    const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
    const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

    const auto* input_weights = (W != nullptr) ? W->Data<float>() : nullptr;
    const auto* recurrent_weights = (R != nullptr) ? R->Data<float>() : nullptr;

    // spans for first direction
    const size_t input_weights_size_per_direction = W_shape[1] * W_shape[2];
    const size_t hidden_weights_size_per_direction = R_shape[1] * R_shape[2];

    GemmWeights<float> W_1(0, input_weights, input_weights_size_per_direction, packed_W_);
    GemmWeights<float> R_1(0, recurrent_weights, hidden_weights_size_per_direction, packed_R_);

    GemmWeights<float> W_2;
    GemmWeights<float> R_2;
    if (direction_ == Direction::kBidirectional) {
      W_2.Init(1, input_weights, input_weights_size_per_direction, packed_W_, nullptr);
      R_2.Init(1, recurrent_weights, hidden_weights_size_per_direction, packed_R_, nullptr);
    }

    return LSTMBase::ComputeImpl<float, float>(*context, W_1, W_2, R_1, R_2);
  } else if (X.IsDataType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
  } else {
    ORT_THROW("Invalid data type for LSTM operator of ", X.DataType());
  }
}

}  // namespace onnxruntime
