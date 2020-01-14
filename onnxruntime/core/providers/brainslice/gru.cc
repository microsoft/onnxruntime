#include "core/providers/brainslice/gru.h"
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "bond_request.h"
#include "bond_response.h"

namespace onnxruntime {
namespace brainslice {

template <>
BrainSliceGRU<float>::BrainSliceGRU(const OpKernelInfo& info) : BrainSliceRNN(info) {
  //0: setup params

  for (auto i = 0; i < num_directions_; ++i) {
    rnn_params_[i].resize(static_cast<uint32_t>(GRUParaIndex::VAR_COUNT));

    auto& inputs = info.node().InputDefs();
    assert(inputs.size() > 0);
    auto input_shape = inputs[0]->Shape();
    ORT_ENFORCE(input_shape, "GRU require input has static shape");
    auto& shape = input_shape->dim();
    assert(shape.size() == 3);
    ORT_ENFORCE(shape[2].dim_param().empty(), "GRU's input dimension need to be static.");
    input_dim_ = static_cast<size_t>(shape[2].dim_value());
	
    //1. convert weights to BrainSliceParameterInitPlan
    std::vector<BrainSliceParameterInitPlan> parameters;
    //a. W
    const Tensor* W;
    ORT_ENFORCE(info.TryGetConstantInput(1, &W), "GRU's W must be a initializers.");
    auto w_dims = W->Shape().GetDims();
    assert(w_dims.size() == 3 && (w_dims[1] % 3) == 0);
    TensorShape w_shape({1, w_dims[1] / 3, w_dims[2]});
    // start position
    char* w_buffer = static_cast<char*>(const_cast<void*>((W->DataRaw()))) + i * 3 * w_shape.Size() * W->DataType()->Size();

    auto matrix_mem_type = use_dram_ ? ISA_Mem_Dram : ISA_Mem_MatrixRf;
    BrainSliceParameterInitPlan wr_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wr_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + w_shape.Size() * W->DataType()->Size(), W->Location());
    BrainSliceParameterInitPlan wz_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wz_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer, W->Location());
    BrainSliceParameterInitPlan wh_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    wh_plan.tensor = onnxruntime::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + 2 * (w_shape.Size() * W->DataType()->Size()), W->Location());

    //b. R
    const Tensor* R;
    ORT_ENFORCE(info.TryGetConstantInput(2, &R), "GRU's R must be a initializers.");
    auto r_dims = R->Shape().GetDims();
    assert(r_dims.size() == 3 && (r_dims[1] % 3) == 0);
    ORT_ENFORCE(hidden_size_ == r_dims[2]);
    TensorShape r_shape({1, r_dims[1] / 3, r_dims[2]});
    char* r_buffer = static_cast<char*>(const_cast<void*>((R->DataRaw()))) + i * 3 * r_shape.Size() * R->DataType()->Size();
    BrainSliceParameterInitPlan rr_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    rr_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + r_shape.Size() * R->DataType()->Size(), R->Location());
    BrainSliceParameterInitPlan rz_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    rz_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer, R->Location());
    BrainSliceParameterInitPlan rh_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    rh_plan.tensor = onnxruntime::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + 2 * (r_shape.Size() * R->DataType()->Size()), R->Location());

    //The built-in GRU firmware is trick that it assume the matrix are load in the order: Wr, Rr, Wz, Rz, Wh, Rh
    //And need to start from address 0. So the order matters here
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_R)], wr_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_R)], rr_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_U)], wz_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_U)], rz_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_C)], wh_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_C)], rh_plan);

    //TODO: share identity matrix for two direction
    //3. upload the identity matrix
    std::vector<float> identity_matrix(hidden_size_ * hidden_size_, 0.0f);
    for (auto j = 0; j < hidden_size_; j++) {
      identity_matrix[j * hidden_size_ + j] = 1.0f;
    }

    BrainSliceParameterInitPlan identity_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 1, false, matrix_mem_type, 0, ISA_Mem_MatrixRf};
    TensorShape identity_shape({hidden_size_, hidden_size_});
    // R is on cpu, reuse the location.
    identity_plan.tensor = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<float>(), identity_shape, &identity_matrix[0], R->Location());
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_IDENTITY)], identity_plan);

    //4. upload bias
    std::vector<std::vector<float>> bias;
    const Tensor* B = nullptr;
    TensorShape b_shape({1, hidden_size_});
    if (info.TryGetConstantInput(3, &B)) {
      auto b_dims = B->Shape().GetDims();
      assert(b_dims.size() == 2 && b_dims[1] % 6 == 0);
      const float* data = B->Data<float>() + i * 6 * b_shape.Size();
      //Wbz:
      std::vector<float> bz(hidden_size_);
      bz.assign(data, data + hidden_size_);
      for (int j = 0; j < hidden_size_; ++j) {
        bz[j] += *(data + 3 * hidden_size_ + j);
      }
      //Wbr:
      std::vector<float> br(hidden_size_);
      br.assign(data + hidden_size_, data + 2 * hidden_size_);
      for (int j = 0; j < hidden_size_; ++j) {
        br[j] += *(data + 4 * hidden_size_ + j);
      }
      //Wbh:
      std::vector<float> bh(hidden_size_);
      bh.assign(data + 2 * hidden_size_, data + 3 * hidden_size_);
      for (int j = 0; j < hidden_size_; ++j) {
        bh[j] += *(data + 5 * hidden_size_ + j);
      }

      bias.push_back(std::move(br));
      bias.push_back(std::move(bz));
      bias.push_back(std::move(bh));
    } else {
      bias.resize(3);
      bias[0].resize(hidden_size_, 0.0f);
      bias[1].resize(hidden_size_, 0.0f);
      bias[2].resize(hidden_size_, 0.0f);
    }

    auto bias_mem_type = use_dram_ ? ISA_Mem_Dram : ISA_Mem_AddSubVrf_0;

    BrainSliceParameterInitPlan br_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    br_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), b_shape, &bias[0][0], W->Location());
    BrainSliceParameterInitPlan bz_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bz_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), b_shape, &bias[1][0], W->Location());
    BrainSliceParameterInitPlan bh_plan = {nullptr, ParameterUsage::USE_AS_VECTOR, 0, false, bias_mem_type, 0, ISA_Mem_AddSubVrf_0};
    bh_plan.tensor = onnxruntime::make_unique<Tensor>(B->DataType(), b_shape, &bias[2][0], W->Location());
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_R)], br_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_U)], bz_plan);
    UploadParameter<float>(&rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_C)], bh_plan);

    // if DRAM mode, assume the kernel own the whole register file and plan the weights
    const BrainSlice_Parameters& bsParameters = provider_->GetFPGAHandle().GetParameters();
    uint32_t outputTiles = (static_cast<uint32_t>(hidden_size_) + bsParameters.NATIVE_DIM - 1) / bsParameters.NATIVE_DIM;
    uint32_t inputTiles = (static_cast<uint32_t>(input_dim_) + bsParameters.NATIVE_DIM - 1) / bsParameters.NATIVE_DIM;
    if (use_dram_) {
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_R)].rfAddress = 0;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_R)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_R)].rfAddress + inputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_U)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_R)].rfAddress + outputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_U)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_U)].rfAddress + inputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_C)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_U)].rfAddress + outputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_C)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WX_C)].rfAddress + inputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_IDENTITY)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MRF_WH_C)].rfAddress + outputTiles * outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_R)].rfAddress = 0;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_U)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_R)].rfAddress + outputTiles;
      rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_C)].rfAddress = rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_U)].rfAddress + outputTiles;
    }

    // plan other temp variables
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MUL_RF_U)] = {0, 0, 0, ISA_Mem_MultiplyVrf};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::MUL_RF_H_PREV)] = {outputTiles, 0, outputTiles, ISA_Mem_MultiplyVrf};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_U)] = {rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_B_C)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_C)] = {rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_U)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_XW_R)] = {rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_C)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_XW_U)] = {rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_XW_R)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
    rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_XW_C)] = {rnn_params_[i][static_cast<uint32_t>(GRUParaIndex::ANS_RF_XW_U)].rfAddress + outputTiles, 0, outputTiles, ISA_Mem_AddSubVrf_0};
  }
}

template <>
Status BrainSliceGRU<float>::Compute(OpKernelContext* context) const {
  auto X = context->Input<Tensor>(0);
  auto& input_shape = X->Shape();
  auto batch_size = input_shape[1];
  if (batch_size != 1)
    return Status(common::ONNXRUNTIME, common::FAIL, "BrainSlice GRU only support batch size 1.");
  //1. prepare the input
  auto* data = X->Data<float>();
  auto seq_len = input_shape[0];
  auto input_dim = input_shape[2];
  std::vector<std::vector<BS_Half>> half_inputs;
  auto step_dim = ((input_dim + native_dim_ - 1) / native_dim_) * native_dim_;
  for (auto i = 0; i < seq_len; ++i) {
    std::vector<BS_Half> half_data;
    for (auto j = 0; j < input_dim; ++j) {
      half_data.push_back(BS_Half(*(data + input_dim * i + j)));
    }
    half_data.resize(step_dim);
    half_inputs.push_back(half_data);
  }

  auto Y = context->Output(0, TensorShape({seq_len, num_directions_, batch_size, hidden_size_}));
  auto Y_h = context->Output(1, TensorShape({num_directions_, batch_size, hidden_size_}));
  if (!Y && !Y_h)  // nothing need to be calculated.
    return Status::OK();

  for (auto i = 0; i < rnn_params_.size(); ++i) {
    Direction direction = (i > 0 || direction_ == Direction::BACKWARD) ? Direction::BACKWARD : Direction::FORWARD;
    assert(input_dim_ == static_cast<uint32_t>(input_dim));
    
	//create a bond parameter
    bool export_hidden = Y != nullptr;
	bond_util::BondStruct eval_args;
    ORT_RETURN_IF_ERROR(CreateEvalBondParameter(seq_len, true, export_hidden, rnn_params_[i], &eval_args));
    
    //since we will only do backward as last step, it is ok to reverse inplace.
    if (direction == Direction::BACKWARD)
      std::reverse(half_inputs.begin(), half_inputs.end());

	size_t pad_output_dim = ((hidden_size_ + native_dim_ - 1) / native_dim_) * native_dim_;
    size_t pad_input_dim = ((input_dim_ + native_dim_ - 1) / native_dim_) * native_dim_;

    const BrainSlice_Parameters& bsParameters = provider_->GetFPGAHandle().GetParameters();
    auto status = provider_->GetFPGAHandle().SendSync(
        [&](void* request, size_t* request_size) {
          void* zero = alloca(pad_output_dim * sizeof(BS_Half));
          memset(zero, 0, pad_output_dim * sizeof(BS_Half));
          
          uint16_t* payloadPtr;
          size_t payloadSize = (pad_output_dim + seq_len * pad_input_dim) * sizeof(BS_Half);
          
          auto status = BrainSlice_Request(&bsParameters, &eval_args, 52, payloadSize, (void**)&payloadPtr, request, request_size);
          if (status)
            return status;

          //copy hist to payload first
          memcpy(payloadPtr, zero, sizeof(BS_Half) * pad_output_dim);
          //copy inputs
          for (auto i = 0; i < half_inputs.size(); ++i) {
            memcpy(payloadPtr + pad_output_dim + i * pad_input_dim, &half_inputs[i][0], sizeof(BS_Half) * pad_input_dim);
          }

          return status;
        },
        [&](const void* response, size_t response_size) {
          size_t payload_size = pad_output_dim * (export_hidden ? seq_len : 1) * sizeof(BS_Half);
          BS_Half* payload;
          auto status = BrainSlice_Response(&bsParameters, response, response_size, (const void**)&payload, &payload_size);
          if (status)
            return status;

          if (export_hidden) {
            auto* y_data = Y->MutableData<float>() + i * hidden_size_;
            assert(payload_size == static_cast<size_t>(seq_len) * pad_output_dim * sizeof(BS_Half));
            for (auto step = 0; step < seq_len; ++step) {
              const BS_Half* output = payload + step * pad_output_dim;
              for (uint32_t j = 0; j < hidden_size_; ++j) {
                *(y_data + j) = *output++;
              }
              y_data += hidden_size_ * (direction_ == Direction::BIDIRECTION ? 2 : 1);
            }
            if (Y_h) {
              auto* y_h_data = Y_h->MutableData<float>() + i * hidden_size_;
              auto* y_last_data = Y->MutableData<float>() + i * hidden_size_ + (seq_len - 1) * hidden_size_ * (direction_ == Direction::BIDIRECTION ? 2 : 1);
              memcpy(y_h_data, y_last_data, sizeof(float) * hidden_size_);
            }
          } else {
            auto y_h_data = Y_h->MutableData<float>() + i * hidden_size_;
            assert(payload_size == pad_output_dim);
            for (uint32_t j = 0; j < hidden_size_; ++j) {
              *(y_h_data + j) = *(payload + j);
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
	GRU,
	kOnnxDomain,
	7,
	kBrainSliceExecutionProvider,
	KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetDefaultInputsMemoryType(OrtMemTypeCPUInput).SetDefaultOutputMemoryType(OrtMemTypeCPUOutput),
	brainslice::BrainSliceGRU<float>);
}  // namespace brainslice
}  // namespace onnxruntime
