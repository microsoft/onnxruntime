# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Shared filtering logic for CUDA contrib ops .cu source lists.
# Both the main CUDA provider and the plugin EP build use identical filtering
# rules for flash attention (quick build) and MoE GEMM FP4/FP8 kernels.
#
# Usage:
#   onnxruntime_filter_cuda_cu_sources(<list_variable_name>)
#
# The macro modifies the named list variable in the caller's scope.

macro(onnxruntime_filter_cuda_cu_sources CU_SRC_LIST)
  # Quick build mode: Filter flash attention kernels for faster development iteration.
  #   - We keep only hdim128 fp16 flash attention kernels in quick build mode.
  #   - All other listed head dimensions are excluded (e.g., 32, 64, 96, 192, 256).
  #     If new head dimensions are added or removed, update this list to match the supported set.
  if(onnxruntime_QUICK_BUILD)
    message(STATUS "Quick build mode enabled: Only building hdim128 fp16 flash attention kernels")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "flash_fwd.*hdim(32|64|96|192|256)")
  endif()

  if(NOT onnxruntime_USE_FP4_QMOE)
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm90_fp4_.*\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_.*\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp4\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
  else()
    # CUDA 13 PTXAS does not complete the FP4 M=128/N=64 pingpong specializations in
    # this build configuration. The dispatcher routes that tile through cooperative
    # mainloop variants instead, so exclude only those unused generated units.
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm90_fp4_(fp16|bf16)_m128_n64_k[0-9]+_cm[12]_cn[12]_pp(_finalize)?\\.generated\\.cu")
  endif()

  if(NOT onnxruntime_USE_FP8_QMOE)
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm90_wfp8_.*\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_fp8_.*\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp8\\.cu")
    list(FILTER ${CU_SRC_LIST} EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
  endif()
endmacro()
