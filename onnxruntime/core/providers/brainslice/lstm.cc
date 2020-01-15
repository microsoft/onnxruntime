
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/lstm.h"
#include "bond_request.h"
#include "bond_response.h"

namespace onnxruntime {
namespace brainslice {

template <>
BrainSliceLSTM<float>::BrainSliceLSTM(const OpKernelInfo& info) : BrainSliceRNN(info) {
  //0: setup params
  for (auto index = 0; index < num_directions_; ++index) {
    rnn_params_[index].resize(static_cast<uint32_t>(LSTMParaIndex::VAR_COUNT));

    //TODO: input_dim need to be know statically
    auto& inputs = info.node().InputDefs();
    assert(inputs.size() > 0);
    auto input_shape = inputs[0]->Shape();
    ORT_ENFORCE(input_shape, "LSTM require input has static shape");
    auto& shape = input_shape->dim();  // [seq_length, batch_size, input_size]
    assert(shape.size() == 3);
    ORT_ENFORCE(shape[2].dim_param().empty(), "LSTM's input dimension need to be static.");
    input_dim_ = static_cast<size_t>(shape[2].dim_value());

    auto matrix_mem_type = use_dram_ ? ISA_Mem_Dram : ISA_Mem_MatrixRf;

    //1. convert weights to BrainSliceParameterInitPlan
    std::vector<BrainSliceParameterInitPlan> parameters;
    //a. W - I, O, F, C
    const Tensor* W;
    ORT_ENFORCE(info.TryGetConstantInput(1, &W), "LSTM's W must be a initializers.");
    auto w_dims = W->Shape().GetDims();
    assert(w_dims.size() == 3 && (w_dims[1] % 4) == 0);
    TensorShape w_shape({1, w_dims[1] / 4, w_dims[2]});
    char* w_buffer = static_cast<char*>(const_cast<void*>((W->DataRaw()))) + index * 4 * w_shape.Size() * W->DataType()->Size();

    BrainSliceParameterInitPlan wi_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wi_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer, W->Location());
    BrainSliceParameterInitPlan wo_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wo_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + w_shape.Size() * W->DataType()->Size(), W->Location());
    BrainSliceParameterInitPlan wf_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wf_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + 2 * (w_shape.Size() * W->DataType()->Size()), W->Location());
    BrainSliceParameterInitPlan wc_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wc_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + 3 * (w_shape.Size() * W->DataType()->Size()), W->Location());

    //b. R - I, O, F, C
    const Tensor* R;
    ORT_ENFORCE(info.TryGetConstantInput(2, &R), "LSTM's R must be a initializers.");
    auto r_dims = R->Shape().GetDims();
    assert(r_dims.size() == 3 && (r_dims[1] % 4) == 0);
    ORT_ENFORCE(hidden_size_ == r_dims[2]);
    TensorShape r_shape({1, r_dims[1] / 4, r_dims[2]});
    char* r_buffer = static_cast<char*>(const_cast<void*>((R->DataRaw()))) + index * 4 * r_shape.Size() * R->DataType()->Size();
    BrainSliceParameterInitPlan ri_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    ri_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer, R->Location());
    BrainSliceParameterInitPlan ro_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    ro_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + r_shape.Size() * R->DataType()->Size(), R->Location());
    BrainSliceParameterInitPlan rf_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    rf_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + 2 * (r_shape.Size() * R->DataType()->Size()), R->Location());
    BrainSliceParameterInitPlan rc_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    rc_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + 3 * (r_shape.Size() * R->DataType()->Size()), R->Location());

    //2. upload the weights
    //The built-in LSTM firmware is trick that it assume the matrix are load in the order: Wi, Ri, Wf, Rf, Wc, Rc, Wo, Ro
    //And need to start from address 0. So the order matters here
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_I)], wi_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_I)], ri_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_F)], wf_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_F)], rf_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_C)], wc_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_C)], rc_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_O)], wo_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_O)], ro_plan);

    //3. upload the identity matrix
    std::vector<float> identity_matrix(hidden_size_ * hidden_size_, 0.0f);
    for (auto i = 0; i < hidden_size_; i++) {
      identity_matrix[i * hidden_size_ + i] = 1.0f;
    }

    BrainSliceParameterInitPlan identity_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 1, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    TensorShape identity_shape({hidden_size_, hidden_size_});
    // R is on cpu, reuse the location.
    identity_plan.tensor = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<float>(), identity_shape, &identity_matrix[0], R->Location());
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_IDENTITY)], identity_plan);

    //4. call LSTM init function
    std::vector<std::vector<float>> bias;
    const Tensor* B = nullptr;  // I, O, F, C
    TensorShape bias_shape({1, hidden_size_});
    if (info.TryGetConstantInput(3, &B)) {
      auto b_dims = B->Shape().GetDims();
      assert(b_dims.size() == 2 && b_dims[1] % 8 == 0);
      const float* data = B->Data<float>() + index * 8 * bias_shape.Size();

      //Wbi:
      std::vector<float> bi(hidden_size_);
      bi.assign(data, data + hidden_size_);
      for (int i = 0; i < hidden_size_; ++i) {
        bi[i] += *(data + 4 * hidden_size_ + i);
      }
      //Wbo:
      std::vector<float> bo(hidden_size_);
      bo.assign(data + hidden_size_, data + 2 * hidden_size_);
      for (int i = 0; i < hidden_size_; ++i) {
        bo[i] += *(data + 5 * hidden_size_ + i);
      }
      //Wbf:
      std::vector<float> bf(hidden_size_);
      bf.assign(data + 2 * hidden_size_, data + 3 * hidden_size_);
      for (int i = 0; i < hidden_size_; ++i) {
        bf[i] += *(data + 6 * hidden_size_ + i);
      }

      //Wbc:
      std::vector<float> bc(hidden_size_);
      bc.assign(data + 3 * hidden_size_, data + 4 * hidden_size_);
      for (int i = 0; i < hidden_size_; ++i) {
        bc[i] += *(data + 7 * hidden_size_ + i);
      }

      bias.push_back(std::move(bi));
      bias.push_back(std::move(bf));
      bias.push_back(std::move(bc));
      bias.push_back(std::move(bo));
    } else {
      bias.resize(4);
      bias[0].resize(hidden_size_, 0.0f);
      bias[1].resize(hidden_size_, 0.0f);
      bias[2].resize(hidden_size_, 0.0f);
      bias[3].resize(hidden_size_, 0.0f);
    }

    auto bias_mem_type = use_dram_ ? ISA_Mem_Dram : ISA_Mem_AddSubVrf_0;

    BrainSliceParameterInitPlan bi_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bi_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), bias_shape, &bias[0][0], W->Location());
    BrainSliceParameterInitPlan bf_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bf_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), bias_shape, &bias[1][0], W->Location());
    BrainSliceParameterInitPlan bc_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bc_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), bias_shape, &bias[2][0], W->Location());
    BrainSliceParameterInitPlan bo_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bo_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), bias_shape, &bias[3][0], W->Location());
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_I)], bi_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_F)], bf_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_C)], bc_plan);
    UploadParameter<float>(&rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_O)], bo_plan);

    // if DRAM mode, assume the kernel own the whole register file and plan the weights
    const BrainSlice_Parameters& bsParameters = provider_->GetFPGAHandle().GetParameters();
    uint32_t outputTiles = (static_cast<uint32_t>(hidden_size_) + bsParameters.NATIVE_DIM - 1) / bsParameters.NATIVE_DIM;
    uint32_t inputTiles = (static_cast<uint32_t>(input_dim_) + bsParameters.NATIVE_DIM - 1) / bsParameters.NATIVE_DIM;
    if (use_dram_) {
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_I)].rfAddress = 0;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_I)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_I)].rfAddress + inputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_F)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_I)].rfAddress + outputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_F)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_F)].rfAddress + inputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_C)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_F)].rfAddress + outputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_C)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_C)].rfAddress + inputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_O)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_C)].rfAddress + outputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_O)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WX_O)].rfAddress + inputTiles * outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_IDENTITY)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MRF_WH_O)].rfAddress + outputTiles * outputTiles;

      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_I)].rfAddress = 0;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_F)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_I)].rfAddress + outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_C)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_F)].rfAddress + outputTiles;
      rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_O)].rfAddress = rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_C)].rfAddress + outputTiles;
    }

    // plan other temp variables
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MUL_RF_I_T)] = {0, 0, 0, ISA_Mem_MultiplyVrf};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MUL_RF_C_PREV)] = {outputTiles, 0, outputTiles, ISA_Mem_MultiplyVrf};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::MUL_RF_C_T_MOD)] = {2 * outputTiles, 0, outputTiles, ISA_Mem_MultiplyVrf};

    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_F_T_MOD)] = {rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_B_O)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_I)] = {rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_F_T_MOD)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_F)] = {rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_I)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_C)] = {rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_F)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_O)] = {rnn_params_[index][static_cast<uint32_t>(LSTMParaIndex::ANS_RF_XW_C)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
  }
}

template <>
Status BrainSliceLSTM<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0); //[seq_length, batch_size, input_size]
  auto& X_shape = X->Shape();
  auto batch_size = X_shape[1];
  if (batch_size != 1)
    return Status(common::ONNXRUNTIME, common::FAIL, "BrainSlice LSTM only support batch size 1.");

  auto sequence_length = X_shape[0];
  auto input_dim = ((X_shape[2] + native_dim_ - 1) / native_dim_) * native_dim_;

  auto* X_data = X->Data<float>();
  std::vector<std::vector<BS_Half>> X_data_half;
  for (auto i = 0; i < sequence_length; ++i) {
    std::vector<BS_Half> half_data;
    for (auto j = 0; j < X_shape[2]; ++j) {
      half_data.push_back(BS_Half(*(X_data + X_shape[2] * i + j)));
    }
    half_data.resize(input_dim);
    X_data_half.push_back(half_data);
  }

  //const Tensor* B = context->Input<Tensor>(3);              // [num_directions, 8*hidden_size]
  //const Tensor* sequence_lens = context->Input<Tensor>(4);  // [batch_size]

  const Tensor* initial_h = context->Input<Tensor>(5);  // initial hidden. [num_directions, batch_size, hidden_size]
  const float* initial_h_data = nullptr;
  std::vector<BS_Half> initial_h_data_half;
  if (initial_h != nullptr) {
    initial_h_data = initial_h->Data<float>();
    for (auto i = 0; i < hidden_size_; ++i) {
      initial_h_data_half.push_back(BS_Half(*(initial_h_data + i)));
    }
  }

  const Tensor* initial_c = context->Input<Tensor>(6);  // initial cell. [num_directions, batch_size, hidden_size]
  const float* initial_c_data = nullptr;
  std::vector<BS_Half> initial_c_data_half;
  if (initial_c != nullptr) {
    initial_c_data = initial_c->Data<float>();
    for (auto i = 0; i < hidden_size_; ++i) {
      initial_c_data_half.push_back(BS_Half(*(initial_c_data + i)));
    }
  }
  //const Tensor* P = context->Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]

  Tensor* Y = context->Output(0, TensorShape({sequence_length, num_directions_, batch_size, hidden_size_}));
  Tensor* Y_h = context->Output(1, TensorShape({num_directions_, batch_size, hidden_size_}));
  Tensor* Y_c = context->Output(2, TensorShape({num_directions_, batch_size, hidden_size_}));
  if (!Y && !Y_h && !Y_c)  // nothing need to be calculated.
    return Status::OK();

  for (auto index = 0; index < num_directions_; ++index) {
    //create a bond parameter
    bool export_hidden = Y != nullptr;
	bond_util::BondStruct eval_args;
    ORT_RETURN_IF_ERROR(CreateEvalBondParameter(sequence_length, true, export_hidden, rnn_params_[index], &eval_args));
	
    if (direction_ == Direction::BACKWARD || index > 0)
      std::reverse(X_data_half.begin(), X_data_half.end());
    
	size_t pad_output_dim = ((hidden_size_ + native_dim_ - 1) / native_dim_) * native_dim_;
    size_t pad_input_dim = ((input_dim_ + native_dim_ - 1) / native_dim_) * native_dim_;

    const BrainSlice_Parameters& bsParameters = provider_->GetFPGAHandle().GetParameters();
    auto status = provider_->GetFPGAHandle().SendSync(
        [&](void* request, size_t* request_size) {
          void* zero = alloca(hidden_size_ * sizeof(BS_Half));
          memset(zero, 0, hidden_size_ * sizeof(BS_Half));
          const void* hist[2];

          hist[0] = initial_h != nullptr ? initial_h_data_half.data() : zero;
          hist[1] = initial_c != nullptr ? initial_c_data_half.data() : zero;

		  uint16_t* payloadPtr;
          size_t payloadSize = (2 * pad_output_dim + sequence_length * pad_input_dim) * sizeof(BS_Half);

          auto status = BrainSlice_Request(&bsParameters, &eval_args, 51, payloadSize, (void**)&payloadPtr, request, request_size);
          if (status)
            return status;
          //copy hist to payload first
          memcpy(payloadPtr, hist[0], sizeof(BS_Half) * pad_output_dim);
          memcpy(payloadPtr + pad_output_dim, hist[1], sizeof(BS_Half) * pad_output_dim);
          //copy inputs
          for (auto i = 0; i < X_data_half.size(); ++i) {
            memcpy(payloadPtr + 2 * pad_output_dim + i * pad_input_dim, &X_data_half[i][0], sizeof(BS_Half) * pad_input_dim);
          }

          return status;
        },
        [&](const void* response, size_t response_size) {
          size_t payload_size = pad_output_dim * (export_hidden ? sequence_length : 1) * sizeof(BS_Half);
          BS_Half* payload;
          auto status = BrainSlice_Response(&bsParameters, response, response_size, (const void**)&payload, &payload_size);
          if (status)
            return status;

          /*size_t output_size, output_count = (eval_args->exportHidden ? eval_args->rnnSteps : 1);
          auto addr_Y = static_cast<const void**>(alloca(output_count * sizeof(const void*)));

          auto status = ONNX_RNN_Functions_EvaluateLstm_Response_Float16(
              &bsParameters,
              eval_args,
              response, response_size,
              addr_Y, &output_size, &output_count);*/

          if (export_hidden) {
            auto* Y_data = Y->MutableData<float>() + index * hidden_size_;
            assert(payload_size == static_cast<size_t>(sequence_length) * pad_output_dim * sizeof(BS_Half));
            for (auto step = 0; step < sequence_length; ++step) {
              const BS_Half* output = payload + step * pad_output_dim;
              for (uint32_t j = 0; j < hidden_size_; ++j) {
                *(Y_data + j) = *output++;
              }
              Y_data += hidden_size_ * (direction_ == Direction::BIDIRECTION ? 2 : 1);
            }
            if (Y_h) {
              auto* Y_h_data = Y_h->MutableData<float>() + index * hidden_size_;
              auto* Y_last_data = Y->MutableData<float>() + index * hidden_size_ + (sequence_length - 1) * hidden_size_ * (direction_ == Direction::BIDIRECTION ? 2 : 1);
              memcpy(Y_h_data, Y_last_data, sizeof(float) * hidden_size_);
            }
          } else {
            auto Y_h_data = Y_h->MutableData<float>() + index * hidden_size_;
            assert(payload_size == pad_output_dim);
            for (auto i = 0; i < hidden_size_; ++i) {
              *(Y_h_data + i) = *(payload + i);
            }
          }

          return status;
        });

    if (!status.IsOK())
      return status;
  }
  return Status::OK();
}
ONNX_OPERATOR_KERNEL_EX(
	LSTM,
	kOnnxDomain,
	7,
	kBrainSliceExecutionProvider,
	KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetDefaultInputsMemoryType(OrtMemTypeCPUInput).SetDefaultOutputMemoryType(OrtMemTypeCPUOutput),
	brainslice::BrainSliceLSTM<float>);
}  // namespace brainslice
}  // namespace onnxruntime
