#ifndef ONNX_RNNS_H_
#define ONNX_RNNS_H_

#include "Firmware/lib/BrainSliceNIOSLib.h"
#include "Onnx_rnns_firmware.h"

typedef struct {
  // InitialVRF (ivrf)
  ISA_VrfAddress ivrf_initial_state;  // initial hidden state, usually zero
                                      //h_prev must be double buffered to avoid an illegal in-place operation
  ISA_VrfAddress ivrf_hr, ivrf_x_active, ivrf_h_prev;
  ISA_VrfAddress ivrf_x_passive, ivrf_h_next;
  ISA_VrfAddress avrf1_rf_c;
} gru_variables_t;

#define MRF_WX_R 0
#define MRF_WH_R 1
#define MRF_WX_U 2
#define MRF_WH_U 3
#define MRF_WX_C 4
#define MRF_WH_C 5

#define MUL_RF_U 6
#define MUL_RF_H_PREV 7

#define ANS_RF_B_R 8
#define ANS_RF_B_U 9
#define ANS_RF_B_C 10
#define ANS_RF_U 11
#define ANS_RF_C 12
#define ANS_RF_XW_R 13
#define ANS_RF_XW_U 14
#define ANS_RF_XW_C 15

#define MRF_IDENTITY 16

typedef struct {
  // InitialVRF (ivrf)
  ISA_VrfAddress ivrf_initial_state;  // initial hidden state, usually zero
                                      //h_prev must be double buffered to avoid an illegal in-place operation
  ISA_VrfAddress ivrf_x_active, ivrf_x_passive, ivrf_h_next, ivrf_h_prev;
  ISA_VrfAddress ivrf_c_prev;  // used to store previous in case it is needed for the output
  ISA_VrfAddress avrf1_rf_c;
} lstm_variables_t;

// MatrixRF (mrf)
#define LSTM_MRF_WX_I 0
#define LSTM_MRF_WH_I 1
#define LSTM_MRF_WX_F 2
#define LSTM_MRF_WH_F 3
#define LSTM_MRF_WX_C 4
#define LSTM_MRF_WH_C 5
#define LSTM_MRF_WX_O 6
#define LSTM_MRF_WH_O 7

// MultiplyVRF (mulvrf)
#define LSTM_MUL_RF_I_T 8
#define LSTM_MUL_RF_C_PREV 9
#define LSTM_MUL_RF_C_T_MOD 10

// AddSubVRF (asvrf)
#define LSTM_ANS_RF_B_I 11
#define LSTM_ANS_RF_B_F 12
#define LSTM_ANS_RF_B_C 13
#define LSTM_ANS_RF_B_O 14
#define LSTM_ANS_RF_F_T_MOD 15
#define LSTM_ANS_RF_XW_I 16
#define LSTM_ANS_RF_XW_F 17
#define LSTM_ANS_RF_XW_C 18
#define LSTM_ANS_RF_XW_O 19
#define LSTM_MRF_IDENTITY 20

#endif /* ONNX_RNNS_H_ */
