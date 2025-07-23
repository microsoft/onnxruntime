// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class LSTMOpBuilder : public BaseOpBuilder {
 public:
  LSTMOpBuilder() : BaseOpBuilder("LSTMOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LSTMOpBuilder);

 protected:
  /*
  ONNX LSTM inputs:
  in[0]: X [seq_length, batch_size, input_size], the input sequences packed
  in[1]: W [num_directions, 4*hidden_size, input_size], the weight tensor for the gates. Concatenation of W[iofc] and WB[iofc]
  in[2]: R [num_directions, 4*hidden_size, hidden_size], the recurrence weight tensor. Concatenation of R[iofc] and RB[iofc]

  ONNX LSTM optional inputs:
  in[3]: B [num_directions, 8*hidden_size], the bias tensor for input gate. Concatenation of [Wb[iofc], Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional)
  in[4]: sequence_lens
  in[5]: initial_h [num_directions, batch_size, hidden_size].
  in[6]: initial_c [num_directions, batch_size, hidden_size].
  in[7]: P [num_directions, 3*hidde_size], the weight tensor for peepholes. Concatenation of P[iof] and PB[iof]

  ONNX LSTM Parameters:
  - activation_alpha ---> Not supported by QNN.
  - activation_beta  ---> Not supported by QNN.
  - activations      ---> Not supported by QNN.
  - clip             ---> Not supported by QNN since the clip in ONNX applied to iofc while QNN only apply to c. Refer
                          https://github.com/microsoft/onnxruntime/blob/v1.21.0/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc
  - direction
  - hidden_size
  - input_forget     ---> Not supported by QNN
  - layout: The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [batch_size, num_directions, hidden_size].

  ONNX LSTM optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size] = stack of out[0] from QNN_LSTM with varient directions
  out[1]: Y_h [num_directions, batch_size, hidden_size] = stack of out[2] from QNN_LSTM with varient directions
  out[2]: Y_c [num_directions, batch_size, hidden_size] = stack of out[1] from QNN_LSTM with varient directions

  QNN LSTM inputs:
  in[0]: x_t: 2D of shape [batch_size, input_size] or
              3D of shape [time_steps, batch_size, input_size] if time_major
                          [batch_size, time_steps, input_size] else
  in[1]: W_xf: input-to-forget weights [num_units, input_size]      = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
  in[2]: W_xc: input-to-cell weights [num_units, input_size]        = ONNX in[1][direction, 3*hidden_size:4*hidden_size, :]
  in[3]: W_xo: input-to-output weights [num_units, input_size]      = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
  in[4]: W_hf: recurrent-to-forget weights [num_units, output_size] = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]
  in[5]: W_hc: recurrent-to-cell weights [num_units, output_size]   = ONNX in[2][direction, 3*hidden_size:4*hidden_size, :]
  in[6]: W_ho: recurrent-to-output weights [num_units, output_size] = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
  in[7]: b_f: forget gate bias [num_units]                          = ONNX in[3][direction, 2*hidden_size:3*hidden_size] + in[3][direction, 6*hidden_size:7*hidden_size]
  in[8]: b_c: cell bias [num_units]                                 = ONNX in[3][direction, 3*hidden_size:4*hidden_size] + in[3][direction, 7*hidden_size:8*hidden_size]
  in[9]: b_o: output gate bias [num_units]                          = ONNX in[3][direction, 1*hidden_size:4*hidden_size] + in[3][direction, 5*hidden_size:6*hidden_size]

  # optional inputs
  in[10]: h_t_init: hidden state init [batch_size, output_size]     = ONNX in[5][direction]
  in[11]: c_t_init: cell state init [batch_size, num_units]         = ONNX in[6][direction]
  in[12]: The input layer normalization weights  ---> not supported on fp16 yet.
  in[13]: The forget layer normalization weights ---> not supported on fp16 yet.
  in[14]: The cell layer normalization weights   ---> not supported on fp16 yet.
  in[15]: The output layer normalization weights ---> not supported on fp16 yet.
  in[16]: W_xi: input-to-input weights [num_units, input_size]      = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
  in[17]: W_hi: recurrent-to-input weights [num_units, output_size] = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
  in[18]: W_ci: cell-to-input weights [num_units]                   = ONNX in[7][direction, 0*hidden_size:1*hidden_size]
  in[19]: W_cf: cell-to-forget weights [num_units]                  = ONNX in[7][direction, 2*hidden_size:3*hidden_size]
  in[20]: W_co: cell-to-output weights [num_units]                  = ONNX in[7][direction, 1*hidden_size:2*hidden_size]
  in[21]: b_i: input gate bias [num_units]                          = ONNX in[3][direction, 0*hidden_size:1*hidden_size] + in[3][direction, 4*hidden_size:5*hidden_size]
  in[22]: W_proj: projection weights [output_size, num_units]     ---> not used
  in[23]: b_proj: projection bias [output_size]                   ---> not used
  in[24]: reset: Determines if the internal state should be reset ---> not used

  QNN LSTM Parameters:
  - direction
  - cell_clip_threshold   ---> not used
  - output_clip_threshold ---> not used
  - time_major
  - input_gate_qscale     ---> not used since we fallback to fp16.
  - forget_gate_qscale    ---> not used since we fallback to fp16.
  - cell_gate_qscale      ---> not used since we fallback to fp16.
  - output_gate_qscale    ---> not used since we fallback to fp16.
  - hidden_state_offset   ---> not used since we fallback to fp16.
 -  hidden_state_qscale   ---> not used since we fallback to fp16.

  QNN LSTM outputs:
  out[0]: h_t 2D of shape [batch_size, output_size] or
              3D of shape [time_steps, batch_size, output_size] if time_major
                          [batch_size, time_steps, output_size] else
  out[1]: c_t [batch_size, num_unit]
  out[2]: o_t [batch_size, output_size]

  QNN LSTM optional outputs:
  out[3]: input_gate [batch_size, num_unit]      ---> not used
  out[4]: forget_gate [batch_size, num_unit]     ---> not used
  out[5]: cell_gate [batch_size, num_unit]       ---> not used
  out[6]: output_gate [batch_size, num_unit]     ---> not used
  out[7]: hidden_state [batch_size, output_size] ---> not used
  */

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const OrtNodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const OrtNodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                             const OrtNodeUnit& node_unit,
                             const std::string& direction,
                             const std::vector<std::string>& input_names,
                             const logging::Logger& logger,
                             const bool& do_op_validation,
                             const bool& is_bidirection,
                             std::vector<std::string>& uni_lstm_output_names) const;
  Status AddStridedSliceOrReshape(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const std::string& input_name,
                                  const std::string& output_name,
                                  const std::vector<uint32_t>& input_shape,
                                  const std::vector<uint32_t>& output_shape,
                                  const std::vector<std::vector<int32_t>>& ranges,
                                  const uint32_t& begin_mask,
                                  const uint32_t& end_mask,
                                  const uint32_t& shrink_axes,
                                  const uint32_t& new_axes_mask,
                                  const Qnn_DataType_t& tensor_data_type,
                                  const QnnQuantParamsWrapper& quantize_param,
                                  bool do_op_validation,
                                  bool is_for_input,
                                  bool is_for_output) const;
};

Status LSTMOpBuilder::AddStridedSliceOrReshape(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit,
                                               const std::string& input_name,
                                               const std::string& output_name,
                                               const std::vector<uint32_t>& input_shape,
                                               const std::vector<uint32_t>& output_shape,
                                               const std::vector<std::vector<int32_t>>& ranges,
                                               const uint32_t& begin_mask,
                                               const uint32_t& end_mask,
                                               const uint32_t& shrink_axes,
                                               const uint32_t& new_axes_mask,
                                               const Qnn_DataType_t& tensor_data_type,
                                               const QnnQuantParamsWrapper& quantize_param,
                                               bool do_op_validation,
                                               bool is_for_input,
                                               bool is_for_output) const {
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_name)) {
    return Status::OK();
  }
  // add strided_slice or reshape
  // this is not general condition, only limited to caller in this builder
  size_t minSize = std::min(input_shape.size(), output_shape.size());
  if (input_shape[0] == 1 && std::equal(output_shape.rbegin(), output_shape.rbegin() + minSize, input_shape.rbegin())) {
    // add Reshape
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_name,
                                                         output_name,
                                                         input_shape,
                                                         output_shape,
                                                         tensor_data_type,
                                                         quantize_param.Copy(),
                                                         quantize_param.Copy(),
                                                         do_op_validation,
                                                         is_for_input,
                                                         is_for_output));
  } else {
    // add StridedSlice
    // inputs
    QnnTensorWrapper input_tensorwrapper(input_name, is_for_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_NATIVE,
                                         tensor_data_type, quantize_param.Copy(),
                                         std::vector<uint32_t>(input_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                      "Failed to add input tensor for inserted StridedSlice or Reshape.");

    // params
    const std::string& node_name = output_name;

    // ranges
    std::vector<uint32_t> ranges_data;
    for (size_t i = 0; i < ranges.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        ranges_data.emplace_back(SafeInt<uint32_t>(ranges[i][j]));
      }
    }
    QnnParamWrapper ranges_param_wrapper(node_unit.Index(), node_name, QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                         {static_cast<uint32_t>(ranges.size()), 3}, std::move(ranges_data), true);
    std::vector<std::string> param_names = {
        ranges_param_wrapper.GetParamTensorName(),
    };
    qnn_model_wrapper.AddParamWrapper(std::move(ranges_param_wrapper));

    // begin_mask
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, begin_mask,
                                               QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, param_names));

    // end_mask
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, end_mask,
                                               QNN_OP_STRIDED_SLICE_PARAM_END_MASK, param_names));

    // shrink_axes
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, shrink_axes,
                                               QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES, param_names));

    // new_axes_mask
    ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_name, new_axes_mask,
                                               QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK, param_names));

    // outputs
    QnnTensorWrapper output_tensorwrapper(output_name,
                                          is_for_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                          tensor_data_type,
                                          quantize_param.Copy(),
                                          std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                      "Failed to add output tensor for inserted StridedSlice.");
    // addNode
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_STRIDED_SLICE, {input_name},
                                                      {output_name}, std::move(param_names), do_op_validation),
                      "Failed to create manually inserted Qnn StridedSlice node.");
  }

  return Status::OK();
}

Status LSTMOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(logger);
  if (node_unit.Inputs().size() > 4 && node_unit.Inputs()[4].Exists()) {
    TensorInfo tensor_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[4], tensor_info));

    ORT_RETURN_IF_NOT(tensor_info.is_initializer, "QNN EP: dynamic sequence_length is not supported.");

    std::vector<uint8_t> sequence_lens_bytes;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*tensor_info.initializer_tensor, sequence_lens_bytes));
    const size_t num_elems = sequence_lens_bytes.size() / sizeof(int32_t);
    gsl::span<const int32_t> sequence_lens{reinterpret_cast<const int32_t*>(sequence_lens_bytes.data()), num_elems};
    ORT_RETURN_IF(std::any_of(sequence_lens.begin(),
                              sequence_lens.end(),
                              [sequence_lens](int i) { return i != sequence_lens[0]; }),
                  "QNN EP: Only support LSTM with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(qnn_model_wrapper.GetOrtApi(), node_unit);
  const float clip = node_helper.Get("clip", (float)0.0);
  ORT_RETURN_IF(clip != 0,
                "QNN EP doesn't support non-default clip for LSTM.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  ORT_RETURN_IF((activations.size() >= 3 && (activations[0] != "sigmoid" || activations[1] != "tanh" || activations[2] != "tanh")) ||
                    (activations.size() == 6 && (activations[3] != "sigmoid" || activations[5] != "tanh" || activations[5] != "tanh")),
                "QNN EP doesn't support non-default activations for LSTM.");
  // TODO: Add support for layout==1
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));
  ORT_RETURN_IF_NOT(layout == 0,
                    "QNN EP: Unsupport layout mode %ld for %s.", layout, node_unit.Name().c_str());
  return Status::OK();
}

Status LSTMOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[i], logger, input_names));
    } else {
      input_names.emplace_back("");
    }
  }
  return Status::OK();
}

Status LSTMOpBuilder::AddUnidirectionLSTM(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const std::string& direction,
                                          const std::vector<std::string>& input_names,
                                          const logging::Logger& logger,
                                          const bool& do_op_validation,
                                          const bool& is_bidirection,
                                          std::vector<std::string>& uni_lstm_output_names) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& onnx_inputs = node_unit.Inputs();
  const auto& onnx_outputs = node_unit.Outputs();
  const std::string& node_name = node_unit.Name();
  std::vector<TensorInfo> input_tensor_infos(onnx_inputs.size());
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_inputs[i], input_tensor_infos[i]));
    }
  }
  // becuase QNN LSTM three outputs are mandatory, we should provide them tensor info
  std::vector<TensorInfo> output_tensor_infos(3);
  for (size_t i = 0; i < 3; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(onnx_outputs[i], output_tensor_infos[i]));
    } else {
      output_tensor_infos[i].qnn_data_type = input_tensor_infos[0].qnn_data_type;
    }
  }

  OrtNodeAttrHelper node_helper(qnn_model_wrapper.GetOrtApi(), node_unit);
  const uint32_t hidden_size = node_helper.Get("hidden_size", 0);
  const int32_t hidden_size_sign = SafeInt<int32_t>(hidden_size);
  ORT_RETURN_IF_NOT(hidden_size > 0, "hidden size is not set for LSTM");
  const int64_t layout = node_helper.Get("layout", static_cast<int64_t>(0));

  const uint32_t input_size = input_tensor_infos[0].shape[2];
  const uint32_t batch_size = layout == 0 ? input_tensor_infos[0].shape[1] : input_tensor_infos[0].shape[0];
  const uint32_t seq_length = layout == 0 ? input_tensor_infos[0].shape[0] : input_tensor_infos[0].shape[1];
  const int32_t direction_idx = input_tensor_infos[1].shape[0] < 2 || direction == "forward" ? 0 : 1;

  // params
  std::vector<std::string> param_names;

  // direction
  ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                             direction == "forward" ? QNN_OP_LSTM_DIRECTION_FORWARD : QNN_OP_LSTM_DIRECTION_REVERSE,
                                             QNN_OP_LSTM_PARAM_DIRECTION, param_names));

  // cell_clip_threshold
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD, param_names));

  // output_clip_threshold
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD, param_names));

  // time_major
  ORT_RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false,
                                         QNN_OP_LSTM_PARAM_TIME_MAJOR, param_names));

  // // input_gate_qscale
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE, param_names));

  // // forget_gate_qscale
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE, param_names));

  // // cell_gate_qscale
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE, param_names));

  // // output_gate_qscale
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE, param_names));

  // // hidden_state_offset
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_HIDDEN_STATE_OFFSET, param_names));

  // // hidden_state_qscale
  ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 0.0,
                                          QNN_OP_LSTM_PARAM_HIDDEN_STATE_QSCALE, param_names));

  // Common LSTM cell inputs
  const std::string null_tensor_name = "null_tensor";
  QnnTensorWrapper null_tensor_wrapper(null_tensor_name, QNN_TENSOR_TYPE_NULL, QNN_DATATYPE_UNDEFINED,
                                       QnnQuantParamsWrapper(), std::vector<uint32_t>{0});

  qnn_model_wrapper.AddTensorWrapper(std::move(null_tensor_wrapper));
  std::vector<std::string> qnn_lstm_input_names(24, null_tensor_name);

  // input W
  {
    // QNN in[1] = ONNX in[1][direction, 2*hidden_size:3*hidden_size, :]
    // QNN in[2] = ONNX in[1][direction, 3*hidden_size:4*hidden_size, :]
    // QNN in[3] = ONNX in[1][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[16] = ONNX in[1][direction, 0*hidden_size:1*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {1, 2, 3, 16};
    std::vector<int32_t> begins = {2, 3, 1, 0};
    std::vector<std::string> qnn_lstm_weight_name = {
        input_names[1] + "_input_to_forget_gate_weight_" + direction,
        input_names[1] + "_input_to_cell_gate_weight_" + direction,
        input_names[1] + "_input_to_output_gate_weight_" + direction,
        input_names[1] + "_input_to_input_gate_weight_" + direction,
    };
    for (size_t i = 0; i < 4; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, SafeInt<int32_t>(input_size), 1}};
      std::vector<uint32_t> output_shape = {hidden_size, input_size};
      ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                   /*node_unit=*/node_unit,
                                                   /*input_name=*/input_names[1],
                                                   /*output_name=*/qnn_lstm_weight_name[i],
                                                   /*input_shape=*/input_tensor_infos[1].shape,
                                                   /*output_shape=*/output_shape,
                                                   /*ranges=*/ranges,
                                                   /*begin_mask=*/begin_mask,
                                                   /*end_mask=*/end_mask,
                                                   /*shrink_axes=*/shrink_axes,
                                                   /*new_axes_mask=*/new_axes_mask,
                                                   /*tensor_data_type=*/input_tensor_infos[1].qnn_data_type,
                                                   /*QnnQuantParamsWrapper=*/input_tensor_infos[1].quant_param,
                                                   /*do_op_validation=*/do_op_validation,
                                                   /*is_for_input=*/false,
                                                   /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input R
  {
    // QNN in[4] = ONNX in[2][direction, 2*hidden_size:3*hidden_size, :]
    // QNN in[5] = ONNX in[2][direction, 3*hidden_size:4*hidden_size, :]
    // QNN in[6] = ONNX in[2][direction, 1*hidden_size:2*hidden_size, :]
    // QNN in[17] = ONNX in[2][direction, 0*hidden_size:1*hidden_size, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<uint32_t> qnn_input_indices = {4, 5, 6, 17};
    std::vector<int32_t> begins = {2, 3, 1, 0};
    std::vector<std::string> qnn_lstm_weight_name = {
        input_names[2] + "_recurrent_to_forget_gate_weight_" + direction,
        input_names[2] + "_recurrent_to_cell_gate_weight_" + direction,
        input_names[2] + "_recurrent_to_output_gate_weight_" + direction,
        input_names[2] + "_recurrent_to_input_gate_weight_" + direction};
    for (size_t i = 0; i < 4; i++) {
      std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                  {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
                                                  {0, hidden_size_sign, 1}};
      std::vector<uint32_t> output_shape = {hidden_size, hidden_size};
      ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                   /*node_unit=*/node_unit,
                                                   /*input_name=*/input_names[2],
                                                   /*output_name=*/qnn_lstm_weight_name[i],
                                                   /*input_shape=*/input_tensor_infos[2].shape,
                                                   /*output_shape=*/output_shape,
                                                   /*ranges=*/ranges,
                                                   /*begin_mask=*/begin_mask,
                                                   /*end_mask=*/end_mask,
                                                   /*shrink_axes=*/shrink_axes,
                                                   /*new_axes_mask=*/new_axes_mask,
                                                   /*tensor_data_type=*/input_tensor_infos[2].qnn_data_type,
                                                   /*QnnQuantParamsWrapper=*/input_tensor_infos[2].quant_param,
                                                   /*do_op_validation=*/do_op_validation,
                                                   /*is_for_input=*/false,
                                                   /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input B
  {
    // QNN in[7] = ONNX in[3][direction, 2*hidden_size:3*hidden_size] + ONNX in[3][direction, 6*hidden_size:7*hidden_size]
    // QNN in[8] = ONNX in[3][direction, 3*hidden_size:4*hidden_size] + ONNX in[3][direction, 7*hidden_size:8*hidden_size]
    // QNN in[9] = ONNX in[3][direction, 1*hidden_size:2*hidden_size] + ONNX in[3][direction, 5*hidden_size:6*hidden_size]
    // QNN in[21] = ONNX in[3][direction, 0*hidden_size:1*hidden_size] + ONNX in[3][direction, 4*hidden_size:5*hidden_size]
    uint32_t begin_mask = 0b00U;
    uint32_t end_mask = 0b00U;
    uint32_t shrink_axes = 0b01U;
    uint32_t new_axes_mask = 0b00U;
    std::vector<uint32_t> output_shape = {hidden_size};
    std::vector<std::string> qnn_lstm_bias_name = {
        node_name + "_forget_gate_bias_" + direction,
        node_name + "_cell_gate_bias_" + direction,
        node_name + "_output_gate_bias_" + direction,
        node_name + "_input_gate_bias_" + direction};
    std::vector<uint32_t> qnn_input_indices = {7, 8, 9, 21};
    if (onnx_inputs.size() > 3 && onnx_inputs[3].Exists()) {
      std::vector<int32_t> begins = {2, 3, 1, 0, 6, 7, 5, 4};
      std::vector<std::string> onnx_lstm_bias_name = {
          input_names[3] + "_input_to_forget_gate_bias_" + direction,
          input_names[3] + "_input_to_cell_gate_bias_" + direction,
          input_names[3] + "_input_to_output_gate_bias_" + direction,
          input_names[3] + "_input_to_input_gate_bias_" + direction,
          input_names[3] + "_recurrent_to_forget_gate_bias_" + direction,
          input_names[3] + "_recurrent_to_cell_gate_bias_" + direction,
          input_names[3] + "_recurrent_to_output_gate_bias_" + direction,
          input_names[3] + "_recurrent_to_input_gate_bias_" + direction};
      for (size_t i = 0; i < 8; i++) {
        std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                    {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1}};
        ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                     /*node_unit=*/node_unit,
                                                     /*input_name=*/input_names[3],
                                                     /*output_name=*/onnx_lstm_bias_name[i],
                                                     /*input_shape=*/input_tensor_infos[3].shape,
                                                     /*output_shape=*/output_shape,
                                                     /*ranges=*/ranges,
                                                     /*begin_mask=*/begin_mask,
                                                     /*end_mask=*/end_mask,
                                                     /*shrink_axes=*/shrink_axes,
                                                     /*new_axes_mask=*/new_axes_mask,
                                                     /*tensor_data_type=*/input_tensor_infos[3].qnn_data_type,
                                                     /*QnnQuantParamsWrapper=*/input_tensor_infos[3].quant_param,
                                                     /*do_op_validation=*/do_op_validation,
                                                     /*is_for_input=*/false,
                                                     /*is_for_output=*/false));
      }
      for (size_t i = 0; i < 4; i++) {
        std::vector<std::string> add_input_names = {onnx_lstm_bias_name[i], onnx_lstm_bias_name[i + 4]};
        // TODO: The quantize_param should not be used directly, we should calculate an approximate quant_param here.
        QnnTensorWrapper add_output_tensorwrapper(qnn_lstm_bias_name[i], QNN_TENSOR_TYPE_NATIVE, input_tensor_infos[3].qnn_data_type,
                                                  input_tensor_infos[3].quant_param.Copy(), std::vector<uint32_t>(output_shape));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_output_tensorwrapper)),
                          "QNN EP: Failed to add output tensor for inserted ElementWiseAdd node.");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD,
                                                          std::move(add_input_names), {qnn_lstm_bias_name[i]}, {}, do_op_validation),
                          "Failed to create manually inserted ElementWiseAdd node.");
        qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_bias_name[i];
      }
    } else {
      // prepare zero bias
      std::string zero_bias_name = node_name + "_zero_bias";
      QnnTensorWrapper zero_bias_tensor_wrapper(zero_bias_name,
                                                QNN_TENSOR_TYPE_STATIC,
                                                input_tensor_infos[0].qnn_data_type,
                                                QnnQuantParamsWrapper(),
                                                std::vector<uint32_t>(output_shape),
                                                std::vector<uint8_t>(
                                                    utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * hidden_size,
                                                    0));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_bias_tensor_wrapper)),
                        "Failed to add additional zero bias for QNN LSTM node.");
      for (size_t i = 0; i < 4; i++) {
        qnn_lstm_input_names[qnn_input_indices[i]] = zero_bias_name;
      }
    }
  }

  // input P
  if (onnx_inputs.size() > 7 && onnx_inputs[7].Exists()) {
    // QNN in[18] = ONNX in[7][direction, 0*hidden_size:1*hidden_size]
    // QNN in[19] = ONNX in[7][direction, 2*hidden_size:1*hidden_size]
    // QNN in[20] = ONNX in[7][direction, 1*hidden_size:1*hidden_size]
    uint32_t begin_mask = 0b00U;
    uint32_t end_mask = 0b00U;
    uint32_t shrink_axes = 0b01U;
    uint32_t new_axes_mask = 0b00U;
    std::vector<uint32_t> output_shape = {hidden_size};
    std::vector<uint32_t> qnn_input_indices = {18, 19, 20};
    std::vector<int32_t> begins = {0, 2, 1};
    std::vector<std::string> qnn_lstm_weight_name = {
        input_names[7] + "_cell_to_input_gate_weight_" + direction,
        input_names[7] + "_cell_to_forget_gate_weight_" + direction,
        input_names[7] + "_cell_to_output_gate_weight_" + direction};
    for (size_t i = 0; i < 3; i++) {
      std::vector<std::vector<int32_t>> ranges = {
          {direction_idx, direction_idx + 1, 1},
          {begins[i] * hidden_size_sign, (begins[i] + 1) * hidden_size_sign, 1},
      };
      ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                   /*node_unit=*/node_unit,
                                                   /*input_name=*/input_names[7],
                                                   /*output_name=*/qnn_lstm_weight_name[i],
                                                   /*input_shape=*/input_tensor_infos[7].shape,
                                                   /*output_shape=*/output_shape,
                                                   /*ranges=*/ranges,
                                                   /*begin_mask=*/begin_mask,
                                                   /*end_mask=*/end_mask,
                                                   /*shrink_axes=*/shrink_axes,
                                                   /*new_axes_mask=*/new_axes_mask,
                                                   /*tensor_data_type=*/input_tensor_infos[7].qnn_data_type,
                                                   /*QnnQuantParamsWrapper=*/input_tensor_infos[7].quant_param,
                                                   /*do_op_validation=*/do_op_validation,
                                                   /*is_for_input=*/false,
                                                   /*is_for_output=*/false));
      qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_weight_name[i];
    }
  }

  // input initial h, c
  {
    // QNN in[10] = ONNX in[5][direction_idx, :, :]
    // QNN in[11] = ONNX in[6][direction_idx, :, :]
    uint32_t begin_mask = 0b000U;
    uint32_t end_mask = 0b000U;
    uint32_t shrink_axes = 0b001U;
    uint32_t new_axes_mask = 0b000U;
    std::vector<std::vector<int32_t>> ranges = {{direction_idx, direction_idx + 1, 1},
                                                {0, SafeInt<int32_t>(batch_size), 1},
                                                {0, hidden_size_sign, 1}};
    std::vector<uint32_t> src_indices = {5, 6};
    std::vector<uint32_t> qnn_input_indices = {10, 11};
    std::vector<uint32_t> output_shape = {batch_size, hidden_size};
    for (size_t i = 0; i < 2; i++) {
      if (onnx_inputs.size() > src_indices[i] && onnx_inputs[src_indices[i]].Exists()) {
        std::string qnn_lstm_input_name = input_names[src_indices[i]] + "_" + direction;
        ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                     /*node_unit=*/node_unit,
                                                     /*input_name=*/input_names[src_indices[i]],
                                                     /*output_name=*/qnn_lstm_input_name,
                                                     /*input_shape=*/input_tensor_infos[src_indices[i]].shape,
                                                     /*output_shape=*/output_shape,
                                                     /*ranges=*/ranges,
                                                     /*begin_mask=*/begin_mask,
                                                     /*end_mask=*/end_mask,
                                                     /*shrink_axes=*/shrink_axes,
                                                     /*new_axes_mask=*/new_axes_mask,
                                                     /*tensor_data_type=*/input_tensor_infos[src_indices[i]].qnn_data_type,
                                                     /*QnnQuantParamsWrapper=*/input_tensor_infos[src_indices[i]].quant_param,
                                                     /*do_op_validation=*/do_op_validation,
                                                     /*is_for_input=*/false,
                                                     /*is_for_output=*/false));
        qnn_lstm_input_names[qnn_input_indices[i]] = qnn_lstm_input_name;
      } else {
        // prepare zero initial values
        std::string zero_initial_values_name = node_name + "_LSTM_initial_values_" + (i == 0 ? "h" : "c");
        QnnTensorWrapper zero_bias_tensor_wrapper(zero_initial_values_name,
                                                  QNN_TENSOR_TYPE_STATIC,
                                                  input_tensor_infos[0].qnn_data_type,
                                                  QnnQuantParamsWrapper(),
                                                  std::vector<uint32_t>(output_shape),
                                                  std::vector<uint8_t>(
                                                      utils::GetElementSizeByType(input_tensor_infos[0].qnn_data_type) * batch_size * hidden_size,
                                                      0));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_bias_tensor_wrapper)),
                          "Failed to add additional initial values for QNN LSTM node.");
        qnn_lstm_input_names[qnn_input_indices[i]] = zero_initial_values_name;
      }
    }
  }

  // add QNN LSTM
  // since HTP doesn't not support 3d yet, add #sequence_length LSTM node
  std::vector<std::string> qnn_all_hidden_state_names;
  qnn_all_hidden_state_names.resize(seq_length);
  for (uint32_t i = 0; i < seq_length; i++) {
    uint32_t sequence_idx = direction == "forward" ? i : seq_length - i - 1;
    // Add LSTM inputs
    std::vector<std::string> qnn_lstm_input_names_i = qnn_lstm_input_names;

    // input X
    {
      // QNN in[0] = ONNX in[0][sequence_idx, :, :]
      uint32_t begin_mask = 0b000U;
      uint32_t end_mask = 0b000U;
      uint32_t shrink_axes = 0b001U;
      uint32_t new_axes_mask = 0b000U;
      std::vector<std::vector<int32_t>> ranges = {{SafeInt<int32_t>(sequence_idx), SafeInt<int32_t>(sequence_idx + 1), 1},
                                                  {0, SafeInt<int32_t>(batch_size), 1},
                                                  {0, SafeInt<int32_t>(input_size), 1}};
      std::string qnn_lstm_input_name = input_names[0] + "_cell_" + std::to_string(sequence_idx) + "_input";
      std::vector<uint32_t> output_shape = {batch_size, input_size};
      ORT_RETURN_IF_ERROR(AddStridedSliceOrReshape(/*qnn_model_wrapper=*/qnn_model_wrapper,
                                                   /*node_unit=*/node_unit,
                                                   /*input_name=*/input_names[0],
                                                   /*output_name=*/qnn_lstm_input_name,
                                                   /*input_shape=*/input_tensor_infos[0].shape,
                                                   /*output_shape=*/output_shape,
                                                   /*ranges=*/ranges,
                                                   /*begin_mask=*/begin_mask,
                                                   /*end_mask=*/end_mask,
                                                   /*shrink_axes=*/shrink_axes,
                                                   /*new_axes_mask=*/new_axes_mask,
                                                   /*tensor_data_type=*/input_tensor_infos[0].qnn_data_type,
                                                   /*QnnQuantParamsWrapper=*/input_tensor_infos[0].quant_param,
                                                   /*do_op_validation=*/do_op_validation,
                                                   /*is_for_input=*/false,
                                                   /*is_for_output=*/false));
      qnn_lstm_input_names_i[0] = qnn_lstm_input_name;
    }

    // outputs
    std::vector<uint32_t> qnn_lstm_output_shape = {batch_size, hidden_size};

    std::vector<std::string> qnn_lstm_output_names = {
        node_name + "_QNN_LSTM_output_all_hidden_state_" + std::to_string(sequence_idx) + "_" + direction,
        node_name + "_QNN_LSTM_output_cell_state_" + std::to_string(sequence_idx) + "_" + direction,
        node_name + "_QNN_LSTM_output_hidden_state_" + std::to_string(sequence_idx) + "_" + direction};
    qnn_lstm_input_names[10] = qnn_lstm_output_names[2];  // update initial_h
    qnn_lstm_input_names[11] = qnn_lstm_output_names[1];  // update initial_c
    qnn_all_hidden_state_names[sequence_idx] = qnn_lstm_output_names[2];

    for (size_t j = 0; j < 3; j++) {
      QnnTensorWrapper output_tensorwrapper(qnn_lstm_output_names[j],
                                            QNN_TENSOR_TYPE_NATIVE,
                                            output_tensor_infos[j].qnn_data_type,
                                            output_tensor_infos[j].quant_param.Copy(),
                                            std::vector<uint32_t>(qnn_lstm_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                        "QNN EP: Failed to add %ldth output tensor for QNN LSTM.", j);
    }
    std::string lstm_node_name = node_name + "_cell_" + std::to_string(sequence_idx) + "_" + direction;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(lstm_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LSTM,
                                                      std::move(qnn_lstm_input_names_i), std::move(qnn_lstm_output_names),
                                                      std::vector<std::string>(param_names), do_op_validation),
                      "QNN EP: Failed to create Qnn LSTM node.");
  }

  // pack all timestamp outputs together for onnx output[0]
  std::string qnn_pack_output_name = node_name + "_QNN_LSTM_output_hidden_state_all_" + direction;

  // add pack for output[0]
  std::vector<std::string> pack_param_names;
  ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), qnn_pack_output_name, 0,
                                             QNN_OP_PACK_PARAM_AXIS, pack_param_names));

  QnnTensorWrapper pack_output_tensorwrapper(qnn_pack_output_name,
                                             QNN_TENSOR_TYPE_NATIVE,
                                             output_tensor_infos[0].qnn_data_type,
                                             output_tensor_infos[0].quant_param.Copy(),
                                             {seq_length, batch_size, hidden_size});
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(pack_output_tensorwrapper)),
                    "QNN EP: Failed to add output tensor for QNN Pack.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(qnn_pack_output_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PACK,
                                                    std::move(qnn_all_hidden_state_names), {qnn_pack_output_name},
                                                    std::move(pack_param_names), do_op_validation),
                    "QNN EP: Failed to create Qnn Pack node.");

  // add reshape for all outputs to align onnx output shape for unidirection
  std::vector<std::string> qnn_reshape_input_names = {
      qnn_pack_output_name,
      qnn_lstm_input_names[10],
      qnn_lstm_input_names[11]};
  std::vector<std::vector<uint32_t>> qnn_lstm_output_shapes = {
      {seq_length, batch_size, hidden_size},
      {batch_size, hidden_size},
      {batch_size, hidden_size}};
  // in the output shapes below, the value of 1 indicates unidirectional
  std::vector<std::vector<uint32_t>> onnx_lstm_output_shapes = {
      {seq_length, 1, batch_size, hidden_size},
      {1, batch_size, hidden_size},
      {1, batch_size, hidden_size}};
  for (size_t i = 0; i < 3; i++) {
    if (onnx_outputs.size() > i && onnx_outputs[i].Exists()) {
      const std::string reshape_output_name = is_bidirection ? qnn_reshape_input_names[i] + "_unsqueeze_" + direction : onnx_outputs[i].name;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(/*input_name=*/qnn_reshape_input_names[i],
                                                           /*output_name=*/reshape_output_name,
                                                           /*input_shape=*/qnn_lstm_output_shapes[i],
                                                           /*output_shape=*/onnx_lstm_output_shapes[i],
                                                           /*tensor_data_type=*/output_tensor_infos[i].qnn_data_type,
                                                           /*quantize_param=*/output_tensor_infos[i].quant_param,
                                                           /*do_op_validation=*/do_op_validation,
                                                           /*is_for_input=*/false,
                                                           /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(reshape_output_name)));
      uni_lstm_output_names.emplace_back(reshape_output_name);
    } else {
      uni_lstm_output_names.emplace_back("");
    }
  }
  return Status::OK();
}

Status LSTMOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& inputs = node_unit.Inputs();

  OrtNodeAttrHelper node_helper(qnn_model_wrapper.GetOrtApi(), node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 8, "LSTM should receive inputs ranging from 3 to 8!");

  if (direction == "bidirectional") {
    std::vector<std::string> uni_lstm_output_names_forward, uni_lstm_output_names_reverse;
    ORT_RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, "forward", input_names, logger, do_op_validation, true,
                                            uni_lstm_output_names_forward));
    ORT_RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, "reverse", input_names, logger, do_op_validation, true,
                                            uni_lstm_output_names_reverse));

    // Concat forward and reverse output
    for (size_t i = 0; i < 3; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string onnx_output_name = node_unit.Outputs()[i].name;

        // param
        std::vector<std::string> concat_param_names;
        ORT_RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), onnx_output_name,
                                                   static_cast<uint32_t>(output_info.shape.size() - 3),
                                                   QNN_OP_CONCAT_PARAM_AXIS, concat_param_names));

        // create tensor and add op
        Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(onnx_output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
        QnnTensorWrapper concat_output_tensorwrapper(onnx_output_name,
                                                     output_tensor_type,
                                                     output_info.qnn_data_type,
                                                     output_info.quant_param.Copy(),
                                                     std::vector<uint32_t>(output_info.shape));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(concat_output_tensorwrapper)),
                          "QNN EP: Failed to add output tensor for QNN Concat.");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name(), QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                          {uni_lstm_output_names_forward[i], uni_lstm_output_names_reverse[i]},
                                                          {onnx_output_name}, std::move(concat_param_names), do_op_validation),
                          "QNN EP: Failed to create Qnn Concat node.");
      }
    }
  } else {
    std::vector<std::string> uni_lstm_output_names;
    ORT_RETURN_IF_ERROR(AddUnidirectionLSTM(qnn_model_wrapper, node_unit, direction, input_names, logger, do_op_validation, false,
                                            uni_lstm_output_names));
  }
  return Status::OK();
}

void CreateLSTMOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LSTMOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
