#pragma once
#include "core/providers/brainslice/brainslice_kernel.h"
#include "core/providers/brainslice/rnn.h"

namespace onnxruntime {
namespace brainslice {

enum class GRUParaIndex {
  MRF_WX_R = 0,
  MRF_WH_R = 1,
  MRF_WX_U = 2,
  MRF_WH_U = 3,
  MRF_WX_C = 4,
  MRF_WH_C = 5,
  MUL_RF_U = 6,
  MUL_RF_H_PREV = 7,
  ANS_RF_B_R = 8,
  ANS_RF_B_U = 9,
  ANS_RF_B_C = 10,
  ANS_RF_U = 11,
  ANS_RF_C = 12,
  ANS_RF_XW_R = 13,
  ANS_RF_XW_U = 14,
  ANS_RF_XW_C = 15,
  MRF_IDENTITY = 16,
  VAR_COUNT = 17
};

template <typename T>
class BrainSliceGRU : public BrainSliceRNN {
 public:
  explicit BrainSliceGRU(const OpKernelInfo& info);
  virtual Status Compute(OpKernelContext* context) const override;
};
}  // namespace brainslice
}  // namespace onnxruntime
