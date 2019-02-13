// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/brainslice/brainslice_kernel.h"
#include "bond_struct.h"

namespace onnxruntime {
namespace brainslice {

enum class Direction {
  FORWARD = 0,
  BACKWARD = 1,
  BIDIRECTION = 2
};

struct ParameterMemLocation {
  uint32_t rfAddress;
  uint32_t dramAddress;
  uint32_t numTiles;
  ISA_Mem memType;
};

class BrainSliceRNN : public BrainSliceOpKernel {
 public:
  static Direction GetDirection(const std::string& direction) {
    if (std::strcmp(direction.c_str(), "forward") == 0) {
      return Direction::FORWARD;
    } else if (std::strcmp(direction.c_str(), "reverse") == 0) {
      return Direction::BACKWARD;
    } else if (std::strcmp(direction.c_str(), "bidirectional") == 0) {
      return Direction::BIDIRECTION;
    } else {
      ORT_THROW("Undefined direction: " + direction);
    }
  }

  explicit BrainSliceRNN(const OpKernelInfo& info);

  template <typename T>
  Status UploadParameter(ParameterMemLocation* rnn_params_ptr, BrainSliceParameterInitPlan& plan);

 protected:
  
  Status CreateEvalBondParameter(int64_t rnn_steps, bool has_init_states, bool export_hidden, const std::vector<ParameterMemLocation>& rnn_parameters, bond_util::BondStruct* out) const;

  const bool use_dram_ = true;
  int64_t input_dim_;
  int64_t hidden_size_;
  Direction direction_;
  int64_t num_directions_;
  
  std::vector<std::vector<ParameterMemLocation> > rnn_params_;
};

}  // namespace brainslice
}  // namespace onnxruntime
