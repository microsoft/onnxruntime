#include "onnx_rnns.h"

VOID onnx_load_matrix(PBS_CONTEXT bs, const ONNX_RNN_VarAddress* args) {
  BSNL_HEX_Assert(args->memType == ISA_Mem_MatrixRf, NIOS_HEX_BML_INVALID_ARG);
  BSNL_HEX_Assert(args->numTiles > 0, NIOS_HEX_BML_INVALID_ARG);

  // If load into DRAM, is dram enabled in bitstream?
  BSNL_HEX_Assert(bs->m_bsParameters.USE_DRAM, NIOS_HEX_DRAM_NOT_SUPPORTED);

  // must fit into memory
  ISA_ExtAddress memSize = bs->m_bsParameters.MVM_MATRIX_RF_SIZE;
  BSNL_HEX_Assert(args->numTiles + args->rfAddress - 1 < memSize,
                  NIOS_HEX_BML_OUT_OF_BOUNDS);
  moveFilterCount128(bs, ISA_Mem_Dram, args->dramAddress, args->memType, args->rfAddress, 1, args->numTiles);
  end_chain(bs);
}

VOID onnx_load_vector(PBS_CONTEXT bs, const ONNX_RNN_VarAddress* args) {
  BSNL_HEX_Assert(args->memType == ISA_Mem_MvmInitialVrf ||
                      args->memType == ISA_Mem_AddSubVrf_0 ||
                      args->memType == ISA_Mem_AddSubVrf_1 ||
                      args->memType == ISA_Mem_MultiplyVrf,
                  NIOS_HEX_BVL_INVALID_ARG);

  BSNL_HEX_Assert(args->numTiles > 0, NIOS_HEX_BVL_OUT_OF_BOUNDS);

  BSNL_HEX_Assert(bs->m_bsParameters.USE_DRAM, NIOS_HEX_DRAM_NOT_SUPPORTED);

  ISA_ExtAddress memEnd = args->numTiles + args->rfAddress - 1;

  // must fit into memory
  BSNL_HEX_Assert(args->memType != ISA_Mem_MvmInitialVrf ||
                  memEnd < bs->m_bsParameters.INITIAL_VRF_SIZE,
                  NIOS_HEX_BVL_OUT_OF_BOUNDS);
  BSNL_HEX_Assert(args->memType != ISA_Mem_AddSubVrf_0 ||
                  memEnd < bs->m_bsParameters.ADDSUB_VRF_0_SIZE,
                  NIOS_HEX_BVL_OUT_OF_BOUNDS);
  BSNL_HEX_Assert(args->memType != ISA_Mem_AddSubVrf_1 ||
                  memEnd < bs->m_bsParameters.ADDSUB_VRF_1_SIZE,
                  NIOS_HEX_BVL_OUT_OF_BOUNDS);
  BSNL_HEX_Assert(args->memType != ISA_Mem_MultiplyVrf ||
                  memEnd < bs->m_bsParameters.MULTIPLY_VRF_SIZE,
                  NIOS_HEX_BVL_OUT_OF_BOUNDS);

  vRead1D(bs, ISA_Mem_Dram, args->dramAddress, args->numTiles);
  v_wr(bs, args->memType, args->rfAddress);
  end_chain(bs);
}

void onnx_gru_SwapBuffers(gru_variables_t* gru) {
  ISA_VrfAddress tmp;

  tmp = gru->ivrf_x_active;
  gru->ivrf_x_active = gru->ivrf_x_passive;
  gru->ivrf_x_passive = tmp;

  tmp = gru->ivrf_h_prev;
  gru->ivrf_h_prev = gru->ivrf_h_next;
  gru->ivrf_h_next = tmp;
}

void onnx_gru_step(PBS_CONTEXT bs, gru_variables_t* gru,
                   const ONNX_RNN_EvalRNNParams* args, DWORD exportH, DWORD prefetch) {
  const ISA_VrfAddress outputTiles = (ISA_VrfAddress)args->outputDim;
  const ISA_VrfAddress inputTiles = (ISA_VrfAddress)args->inputDim;

  SetIterationsCols(bs, 1, inputTiles);
  // xWr = x * Wr + br
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[MRF_WX_R].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_B_R].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_R].rfAddress + row);
  }

  //xWu = x * Wu + bu
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[MRF_WX_U].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_B_U].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_U].rfAddress + row);
  }

  SetIterationsCols(bs, 1, outputTiles);
  // rh = sigmoid(h * Wr + xWr) * h
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[MRF_WH_R].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_R].rfAddress + row);
    v_sigm(bs);
    vv_mul(bs, args->var_addr_map[MUL_RF_H_PREV].rfAddress + row);
    v_wr(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_hr + row);
  }

  SetIterationsCols(bs, 1, inputTiles);
  // xWc = x * Wc + bc
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[MRF_WX_C].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_B_C].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_C].rfAddress + row);
  }

  SetIterationsCols(bs, 1, outputTiles);
  // u = sigmoid(h * Wu + xWu)
  // save u * h and (1 - u) to MFU1
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[MRF_WH_U].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_U].rfAddress + row);
    v_sigm(bs);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[MUL_RF_U].rfAddress + row);
  }

  // Prefetch next input
  if (prefetch) {
    SetIterationsCols(bs, 1, inputTiles);
    v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
    v_wr(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_x_passive);
  }

  // c = tanh(rh * Wh_c + xWc)
  SetIterationsCols(bs, 1, outputTiles);
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_hr);
    mv_mul(bs, args->var_addr_map[MRF_WH_C].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_XW_C].rfAddress + row);
    v_tanh(bs);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_C].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_1, gru->avrf1_rf_c + row);
  }

  //// h_next = u * h + (1 - u) * c
  ////        = u * h + c - u * c
  ////        = (h - c) * u + c
  SetIterationsCols(bs, 1, outputTiles);
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[MRF_IDENTITY].rfAddress + row * outputTiles);
    vv_a_sub_b_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[ANS_RF_C].rfAddress + row);
    vv_mul(bs, args->var_addr_map[MUL_RF_U].rfAddress + row);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_1, gru->avrf1_rf_c + row);
    v_wr(bs, ISA_Mem_MvmInitialVrf, gru->ivrf_h_next + row);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[MUL_RF_H_PREV].rfAddress + row);
    if (exportH) {
      v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
    }
  }
}

VOID ONNX_RNN_Functions_EvaluateGru(PBS_CONTEXT bs,
                                    const ONNX_RNN_EvalRNNParams* args) {
  const ISA_ScalarValue inputTiles = (ISA_ScalarValue)args->inputDim;
  const ISA_ScalarValue outputTiles = (ISA_ScalarValue)args->outputDim;

  const DWORD rnnSteps = args->rnnSteps;
  const DWORD exportHidden = args->exportHidden;
  const DWORD initHist = args->initHist;
  // Check we have initialized and params are valid
  BSNL_HEX_Assert(inputTiles > 0 && outputTiles > 0, NIOS_HEX_BLE_INVALID_ARG);
  BSNL_HEX_Assert(rnnSteps > 0, NIOS_HEX_BLE_INVALID_ARG);

  ONNX_RNN_Functions_EvaluateGru_PostResponse(bs);

  gru_variables_t gru;
  // InitialVRF
  //   ivrf_initial_state
  //   ivrf_hr
  //   ivrf_x_active
  //   ivrf_h_prev
  //   ivrf_x_passive
  //   ivrf_h_next
  //
  // Set up the double buffers. Read previous timestep's state from one set of
  // buffers and write the current timestep's state outputs to the other set of
  // buffers. Then swap the buffers at the end of each timestep.
  BSNL_HEX_Assert(4 * outputTiles + 2 * inputTiles <
                      bs->m_bsParameters.INITIAL_VRF_SIZE,
                  NIOS_HEX_GRU_INVALID_ARG);
  gru.ivrf_initial_state = 0;

  // [h*r, x, h]
  gru.ivrf_hr = gru.ivrf_initial_state + outputTiles;
  gru.ivrf_x_active = gru.ivrf_hr + outputTiles;
  gru.ivrf_h_prev = gru.ivrf_x_active + inputTiles;

  gru.ivrf_x_passive = gru.ivrf_h_prev + outputTiles;
  gru.ivrf_h_next = gru.ivrf_x_passive + inputTiles;
  gru.avrf1_rf_c = 0;

  // Initialize h_prev into the active side of the double buffer
  SetIterationsCols(bs, 1, outputTiles);
  if (initHist) {
    v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
    v_wr(bs, ISA_Mem_MvmInitialVrf, gru.ivrf_h_prev);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[MUL_RF_H_PREV].rfAddress);
  } else {
    v_rd(bs, ISA_Mem_MvmInitialVrf, gru.ivrf_initial_state);
    v_wr(bs, ISA_Mem_MvmInitialVrf, gru.ivrf_h_prev);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[MUL_RF_H_PREV].rfAddress);
  }

  //load weights from dram if needed
  for (uint32_t i = 0; i < args->var_addr_map_count; ++i) {
    if (args->var_addr_map[i].dramAddress != 0) {
      if (args->var_addr_map[i].memType == ISA_Mem_MatrixRf)
        onnx_load_matrix(bs, &args->var_addr_map[i]);
      else
        onnx_load_vector(bs, &args->var_addr_map[i]);
    }
  }

  // gather input
  SetIterationsCols(bs, 1, inputTiles);
  v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
  v_wr(bs, ISA_Mem_MvmInitialVrf, gru.ivrf_x_active);
  for (DWORD t = 0; t < rnnSteps; t++) {
    onnx_gru_step(bs, &gru, args, exportHidden || t == rnnSteps - 1, t + 1 < rnnSteps);
    onnx_gru_SwapBuffers(&gru);
  }
  end_chain(bs);
}

void onnx_lstm_SwapBuffers(lstm_variables_t* lstm) {
  ISA_VrfAddress tmp;

  tmp = lstm->ivrf_x_active;
  lstm->ivrf_x_active = lstm->ivrf_x_passive;
  lstm->ivrf_x_passive = tmp;

  tmp = lstm->ivrf_h_prev;
  lstm->ivrf_h_prev = lstm->ivrf_h_next;
  lstm->ivrf_h_next = tmp;
}

void onnx_lstm_Step(PBS_CONTEXT bs, lstm_variables_t* lstm,
                    const ONNX_RNN_EvalRNNParams* args, DWORD exportH, DWORD prefetch) {
  const ISA_VrfAddress outputTiles = (ISA_VrfAddress)args->outputDim;
  const ISA_VrfAddress inputTiles = (ISA_VrfAddress)args->inputDim;

  // Compute mvmuls for input
  SetIterationsCols(bs, 1, inputTiles);

  // xWi = x * Wi + bi
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WX_I].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_B_I].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_I].rfAddress + row);
  }
  // xWf = x * Wf + bf
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WX_F].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_B_F].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_F].rfAddress + row);
  }
  // xWc = x * Wc + bc
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WX_C].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_B_C].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_C].rfAddress + row);
  }

  // Prefetch next input
  if (prefetch) {
    SetIterationsCols(bs, 1, inputTiles);
    v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
    v_wr(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_x_passive);
  }

  // Compute gates
  SetIterationsCols(bs, 1, outputTiles);

  // i gate
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WH_I].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_I].rfAddress + row);
    v_sigm(bs);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_MUL_RF_I_T].rfAddress + row);
  }

  // f gate
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WH_F].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_F].rfAddress + row);
    v_sigm(bs);
    vv_mul(bs, args->var_addr_map[LSTM_MUL_RF_C_PREV].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_F_T_MOD].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_1, lstm->avrf1_rf_c + row);
  }
  // c gate
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WH_C].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_C].rfAddress + row);
    v_tanh(bs);
    vv_mul(bs, args->var_addr_map[LSTM_MUL_RF_I_T].rfAddress + row);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_1, lstm->avrf1_rf_c + row);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[LSTM_MUL_RF_C_PREV].rfAddress + row);
    v_wr(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_c_prev);
  }

  SetIterationsCols(bs, 1, outputTiles);
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_c_prev);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_IDENTITY].rfAddress + row * outputTiles);
    v_tanh(bs);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[LSTM_MUL_RF_C_T_MOD].rfAddress + row);
  }

  // xWf = x * Wo + bo
  SetIterationsCols(bs, 1, inputTiles);
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_x_active);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WX_O].rfAddress + row * inputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_B_O].rfAddress + row);
    v_wr(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_O].rfAddress + row);
  }

  // o gate
  SetIterationsCols(bs, 1, outputTiles);
  for (ISA_ExtAddress row = 0; row < outputTiles; row++) {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_h_prev);
    mv_mul(bs, args->var_addr_map[LSTM_MRF_WH_O].rfAddress + row * outputTiles);
    vv_add_mi(bs, ISA_Mem_AddSubVrf_0, args->var_addr_map[LSTM_ANS_RF_XW_O].rfAddress + row);
    v_sigm(bs);
    vv_mul(bs, args->var_addr_map[LSTM_MUL_RF_C_T_MOD].rfAddress + row);
    v_wr(bs, ISA_Mem_MvmInitialVrf, lstm->ivrf_h_next + row);
    if (exportH) {
      v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
    }
  }
}

VOID ONNX_RNN_Functions_EvaluateLstm(PBS_CONTEXT bs, const ONNX_RNN_EvalRNNParams* args) {
  const ISA_ScalarValue inputTiles = (ISA_ScalarValue)args->inputDim;
  const ISA_ScalarValue outputTiles = (ISA_ScalarValue)args->outputDim;

  const DWORD rnnSteps = args->rnnSteps;
  const DWORD initHist = args->initHist;
  const DWORD exportHidden = args->exportHidden;
  // Check we have initialized and params are valid
  BSNL_HEX_Assert(inputTiles > 0 && outputTiles > 0, NIOS_HEX_BLE_INVALID_ARG);
  BSNL_HEX_Assert(rnnSteps > 0, NIOS_HEX_BLE_INVALID_ARG);

  ONNX_RNN_Functions_EvaluateLstm_PostResponse(bs);

  // InitialVRF
  lstm_variables_t lstm;
  lstm.ivrf_initial_state = 0;
  lstm.ivrf_c_prev = lstm.ivrf_initial_state + outputTiles;
  // Set up the double buffers. Read previous timestep's state from one set of buffers and write
  // the current timestep's state outputs to the other set of buffers. Then swap the buffers at the
  // end of each timestep.
  lstm.ivrf_x_active = lstm.ivrf_c_prev + outputTiles;
  lstm.ivrf_h_prev = lstm.ivrf_x_active + inputTiles;
  lstm.ivrf_x_passive = lstm.ivrf_h_prev + outputTiles;
  lstm.ivrf_h_next = lstm.ivrf_x_passive + inputTiles;
  lstm.avrf1_rf_c = 0;

  // Initialize h_prev and c_prev Write h_prev into the active side of the double buffer
  SetIterationsCols(bs, 1, outputTiles);
  if (initHist) {
    v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
    v_wr(bs, ISA_Mem_MvmInitialVrf, lstm.ivrf_h_prev);

    v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[LSTM_MUL_RF_C_PREV].rfAddress);
  } else {
    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm.ivrf_initial_state);
    v_wr(bs, ISA_Mem_MvmInitialVrf, lstm.ivrf_h_prev);

    v_rd(bs, ISA_Mem_MvmInitialVrf, lstm.ivrf_c_prev);
    v_wr(bs, ISA_Mem_MultiplyVrf, args->var_addr_map[LSTM_MUL_RF_C_PREV].rfAddress);
  }

  //load weights from dram if needed
  for (uint32_t i = 0; i < args->var_addr_map_count; ++i) {
    if (args->var_addr_map[i].dramAddress != 0) {
      if (args->var_addr_map[i].memType == ISA_Mem_MatrixRf)
        onnx_load_matrix(bs, &args->var_addr_map[i]);
      else
        onnx_load_vector(bs, &args->var_addr_map[i]);
    }
  }

  // gather input
  SetIterationsCols(bs, 1, inputTiles);
  v_rd(bs, ISA_Mem_NetInputQ, DONTCARE);
  v_wr(bs, ISA_Mem_MvmInitialVrf, lstm.ivrf_x_active);
  for (DWORD t = 0; t < rnnSteps; t++) {
    onnx_lstm_Step(bs, &lstm, args, exportHidden || t == rnnSteps - 1, t + 1 < rnnSteps);
    onnx_lstm_SwapBuffers(&lstm);
  }
  end_chain(bs);
}