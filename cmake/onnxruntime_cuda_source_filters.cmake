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

# Extract SM90/SM120 TMA warp-specialized generated source files from a CUDA source list.
# These files use CUTLASS 3.x features (GMMA, TMA) that are specific to SM90+ or SM120+.
# They are compiled in separate OBJECT libraries with restricted CUDA_ARCHITECTURES to:
#   1. Reduce compile time (avoid compiling heavy templates for unused architectures)
#   2. Reduce binary size (no dead device code for unsupported architectures)
#   3. Ensure correctness (SM90 code compiled at exactly 90a-real, SM120 at 120+)
#
# The per-source CUDA_ARCHITECTURES property does not work with the Visual Studio generator,
# so OBJECT libraries are needed.
#
# Usage:
#   onnxruntime_extract_sm_specific_cuda_sources(<cu_src_list_var>
#       SM90_SOURCES <output_var> SM120_SOURCES <output_var>)
#
# Removes matched files from <cu_src_list_var> and stores them in the output variables.
macro(onnxruntime_extract_sm_specific_cuda_sources CU_SRC_LIST)
  cmake_parse_arguments(_EXTRACT "" "SM90_SOURCES;SM120_SOURCES" "" ${ARGN})

  # Extract SM90 TMA WS generated files
  set(${_EXTRACT_SM90_SOURCES})
  if(ORT_HAS_SM90_OR_LATER)
    foreach(_src IN LISTS ${CU_SRC_LIST})
      if(_src MATCHES "moe_gemm_tma_ws_sm90_.*\\.generated\\.cu$")
        list(APPEND ${_EXTRACT_SM90_SOURCES} "${_src}")
      endif()
    endforeach()
    if(${_EXTRACT_SM90_SOURCES})
      list(REMOVE_ITEM ${CU_SRC_LIST} ${${_EXTRACT_SM90_SOURCES}})
    endif()
  endif()

  # Extract SM120 TMA WS generated files
  set(${_EXTRACT_SM120_SOURCES})
  if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    foreach(_src IN LISTS ${CU_SRC_LIST})
      if(_src MATCHES "moe_gemm_tma_ws_sm120_.*\\.generated\\.cu$")
        list(APPEND ${_EXTRACT_SM120_SOURCES} "${_src}")
      endif()
    endforeach()
    if(${_EXTRACT_SM120_SOURCES})
      list(REMOVE_ITEM ${CU_SRC_LIST} ${${_EXTRACT_SM120_SOURCES}})
    endif()
  endif()
endmacro()
