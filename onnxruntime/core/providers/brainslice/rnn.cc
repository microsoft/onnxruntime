#include "core/providers/brainslice/brainslice_kernel.h"
#include "core/providers/brainslice/rnn.h"

namespace onnxruntime {
namespace brainslice {

BrainSliceRNN::BrainSliceRNN(const OpKernelInfo& info) : BrainSliceOpKernel(info) {
  ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK());

  std::string direction;
  ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());
  direction_ = GetDirection(direction);
  num_directions_ = direction_ == Direction::BIDIRECTION ? 2 : 1;
  rnn_params_.resize(num_directions_);
}

template <>
Status BrainSliceRNN::UploadParameter<float>(ParameterMemLocation* rnn_params_ptr, BrainSliceParameterInitPlan& plan) {
  *rnn_params_ptr = {0, 0, 0, plan.target_mem_type};
  ORT_ENFORCE(BrainSliceOpKernel::UploadBrainSliceParameter<float>(plan, provider_, &rnn_params_ptr->numTiles).IsOK());
  if (plan.mem_type == ISA_Mem_Dram)
    rnn_params_ptr->dramAddress = static_cast<uint32_t>(plan.address);
  else
    rnn_params_ptr->rfAddress = static_cast<uint32_t>(plan.address);
  return Status::OK();
}

Status BrainSliceRNN::CreateEvalBondParameter(int64_t rnn_steps, bool has_init_states, bool export_hidden, const std::vector<ParameterMemLocation>& rnn_parameters, bond_util::BondStruct* out) const {
  std::vector<bond_util::BondStruct> var_addr_map(rnn_parameters.size());
  for (size_t i = 0; i < rnn_parameters.size(); ++i)
    var_addr_map[i] = bond_util::BondStruct({
        {{"rfAddress", 0}, {}, bond_util::Value(rnn_parameters[i].rfAddress)},
        {{"dramAddress", 1}, {}, bond_util::Value(rnn_parameters[i].dramAddress)},
        {{"numTiles", 2}, {}, bond_util::Value(rnn_parameters[i].numTiles)},
        {{"memType", 3}, {}, bond_util::Value(rnn_parameters[i].memType)},
    });

  *out = bond_util::BondStruct({
      {{"rnnSteps", 0}, {}, bond_util::Value(static_cast<uint32_t>(rnn_steps))},
      {{"initHist", 1}, {}, bond_util::Value(static_cast<uint32_t>(has_init_states))},
      {{"exportHidden", 2}, {}, bond_util::Value(static_cast<uint32_t>(export_hidden))},
      {{"inputDim", 3}, {{"TensorDimension", ""}}, bond_util::Value(static_cast<uint32_t>(((input_dim_ + native_dim_ - 1) / native_dim_) * native_dim_))},
      {{"outputDim", 4}, {{"TensorDimension", ""}}, bond_util::Value(static_cast<uint32_t>(((hidden_size_ + native_dim_ - 1) / native_dim_) * native_dim_))},
      {{"var_addr_map", 5}, {}, bond_util::Value(std::move(var_addr_map))},
  });
  return Status::OK();
}

}  // namespace brainslice
}  // namespace onnxruntime
