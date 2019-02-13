#pragma once

#include "core/providers/brainslice/brainslice_kernel.h"
#include "core/providers/brainslice/rnn.h"

namespace onnxruntime {
  namespace brainslice {
    enum class LSTMParaIndex {
      // MatrixRF (mrf)
      MRF_WX_I = 0,
      MRF_WH_I = 1,
      MRF_WX_F = 2,
      MRF_WH_F = 3,
      MRF_WX_C = 4,
      MRF_WH_C = 5,
      MRF_WX_O = 6,
      MRF_WH_O = 7,

      // MultiplyVRF (mulvrf)
      MUL_RF_I_T = 8,
      MUL_RF_C_PREV = 9,
      MUL_RF_C_T_MOD = 10,

      // AddSubVRF (asvrf)
      ANS_RF_B_I = 11,
      ANS_RF_B_F = 12,
      ANS_RF_B_C = 13,
      ANS_RF_B_O = 14,
      ANS_RF_F_T_MOD = 15,
      ANS_RF_XW_I = 16,
      ANS_RF_XW_F = 17,
      ANS_RF_XW_C = 18,
      ANS_RF_XW_O = 19,
      MRF_IDENTITY = 20,
      VAR_COUNT = 21
    };

    template <typename T>
    class BrainSliceLSTM : public BrainSliceRNN {
    public:
      explicit BrainSliceLSTM(const OpKernelInfo& info);
      virtual Status Compute(OpKernelContext* context) const override;
    };
  }  // namespace brainslice
}  // namespace onnxruntime
