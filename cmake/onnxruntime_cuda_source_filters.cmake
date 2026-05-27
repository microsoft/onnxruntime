# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Shared filtering logic for CUDA contrib ops .cu source lists.
# Both the main CUDA provider and the plugin EP build use identical filtering
# rules for flash attention (quick build) and MoE GEMM FP4/FP8 kernels.
#
# Usage:
#   onnxruntime_filter_cuda_cu_sources(<list_variable_name>)
#
# The function modifies the named list variable in the caller's scope.

function(onnxruntime_filter_cuda_cu_sources CU_SRC_LIST)
  set(_list "${${CU_SRC_LIST}}")

  # Quick build mode: Filter flash attention kernels for faster development iteration.
  #   - We keep only hdim128 fp16 and bf16 flash attention kernels in quick build mode.
  #   - All other listed head dimensions are excluded (e.g., 32, 64, 96, 192, 256).
  #     If new head dimensions are added or removed, update this list to match the supported set.
  if(onnxruntime_QUICK_BUILD)
    message(STATUS "Quick build mode enabled: Only building hdim128 fp16/bf16 flash attention kernels")
    list(FILTER _list EXCLUDE REGEX "flash_fwd.*hdim(32|64|96|192|256)")
  endif()

  if(NOT onnxruntime_USE_FP4_QMOE)
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm90_fp4_.*\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_.*\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp4\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
  else()
    # CUDA 13 PTXAS does not complete the FP4 M=128/N=64 pingpong specializations in
    # this build configuration. The dispatcher routes that tile through cooperative
    # mainloop variants instead, so exclude only those unused generated units.
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm90_fp4_(fp16|bf16)_m128_n64_k[0-9]+_cm[12]_cn[12]_pp(_finalize)?\\.generated\\.cu")
  endif()

  if(NOT onnxruntime_USE_FP8_QMOE)
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm90_wfp8_.*\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp4_fp8_.*\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_tma_ws_sm120_fp8_fp4\\.generated\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_kernels_(fp16|bf16)_fp8\\.cu")
    list(FILTER _list EXCLUDE REGEX "moe_gemm_kernels_fp8_fp4\\.cu")
  endif()

  set("${CU_SRC_LIST}" "${_list}" PARENT_SCOPE)
endfunction()

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
function(onnxruntime_extract_sm_specific_cuda_sources CU_SRC_LIST)
  cmake_parse_arguments(PARSE_ARGV 1 _EXTRACT "" "SM90_SOURCES;SM120_SOURCES" "")

  set(_list "${${CU_SRC_LIST}}")

  # Extract SM90 TMA WS generated files
  set(_sm90_srcs)
  if(ORT_HAS_SM90_OR_LATER)
    foreach(_src IN LISTS _list)
      if(_src MATCHES "moe_gemm_tma_ws_sm90_.*\\.generated\\.cu$")
        list(APPEND _sm90_srcs "${_src}")
      endif()
    endforeach()
    if(_sm90_srcs)
      list(REMOVE_ITEM _list ${_sm90_srcs})
    endif()
  endif()

  # Extract SM120 TMA WS generated files
  set(_sm120_srcs)
  if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    foreach(_src IN LISTS _list)
      if(_src MATCHES "moe_gemm_tma_ws_sm120_.*\\.generated\\.cu$")
        list(APPEND _sm120_srcs "${_src}")
      endif()
    endforeach()
    if(_sm120_srcs)
      list(REMOVE_ITEM _list ${_sm120_srcs})
    endif()
  endif()

  set("${CU_SRC_LIST}" "${_list}" PARENT_SCOPE)
  set("${_EXTRACT_SM90_SOURCES}" "${_sm90_srcs}" PARENT_SCOPE)
  set("${_EXTRACT_SM120_SOURCES}" "${_sm120_srcs}" PARENT_SCOPE)
endfunction()

# Extract Flash Attention CUDA source files into a separate list for compilation
# in a dedicated OBJECT library with SM80+ architectures and independent nvcc_threads.
# Flash Attention V2 kernels require SM80 (Ampere) or later — they contain
# __CUDA_ARCH__ >= 800 guards in kernel_traits.h and all files are *_sm80.cu.
# Compiling them separately allows:
#   1. Restricting CUDA_ARCHITECTURES to SM80+ (skip dead pre-Ampere passes)
#   2. Using --threads 1 (memory-intensive) while other targets use higher parallelism
#
# Usage:
#   onnxruntime_extract_flash_attention_sources(<cu_src_list_var>
#       FLASH_SOURCES <output_var>)
function(onnxruntime_extract_flash_attention_sources CU_SRC_LIST)
  cmake_parse_arguments(PARSE_ARGV 1 _FA "" "FLASH_SOURCES" "")

  set(_list "${${CU_SRC_LIST}}")
  set(_flash_srcs)
  foreach(_src IN LISTS _list)
    if(_src MATCHES "/bert/flash_attention/.*\\.cu$")
      list(APPEND _flash_srcs "${_src}")
    endif()
  endforeach()
  if(_flash_srcs)
    list(REMOVE_ITEM _list ${_flash_srcs})
  endif()

  set("${CU_SRC_LIST}" "${_list}" PARENT_SCOPE)
  set("${_FA_FLASH_SOURCES}" "${_flash_srcs}" PARENT_SCOPE)
endfunction()

# Extract LLM CUDA source files into separate lists for per-architecture compilation.
# The LLM directory (contrib_ops/cuda/llm/) contains kernels with minimum SM75 support
# (fpA_intB_gemv/gemm enforce arch >= 75). SM90-specific launchers (fpA_intB_gemm
# launchers guarded by #ifndef EXCLUDE_SM_90) are extracted separately to be compiled
# at 90a-real (merged into the SM90 TMA OBJECT library).
#
# Note: SM90 TMA MoE GEMM files are already extracted by
# onnxruntime_extract_sm_specific_cuda_sources() before this function is called.
#
# Usage:
#   onnxruntime_extract_llm_sources(<cu_src_list_var>
#       LLM_SOURCES <output_var>
#       LLM_SM90_SOURCES <output_var>)
function(onnxruntime_extract_llm_sources CU_SRC_LIST)
  cmake_parse_arguments(PARSE_ARGV 1 _LLM "" "LLM_SOURCES;LLM_SM90_SOURCES" "")

  set(_list "${${CU_SRC_LIST}}")
  set(_llm_srcs)
  set(_llm_sm90_srcs)
  foreach(_src IN LISTS _list)
    if(_src MATCHES "/contrib_ops/cuda/llm/.*\\.cu$")
      # SM90-specific fpA_intB launchers (guarded by #ifndef EXCLUDE_SM_90)
      if(_src MATCHES "fpA_intB_gemm_launcher_[0-9]+\\.generated\\.cu$")
        list(APPEND _llm_sm90_srcs "${_src}")
      else()
        list(APPEND _llm_srcs "${_src}")
      endif()
    endif()
  endforeach()
  if(_llm_srcs)
    list(REMOVE_ITEM _list ${_llm_srcs})
  endif()
  if(_llm_sm90_srcs)
    list(REMOVE_ITEM _list ${_llm_sm90_srcs})
  endif()

  set("${CU_SRC_LIST}" "${_list}" PARENT_SCOPE)
  set("${_LLM_LLM_SOURCES}" "${_llm_srcs}" PARENT_SCOPE)
  set("${_LLM_LLM_SM90_SOURCES}" "${_llm_sm90_srcs}" PARENT_SCOPE)
endfunction()

# Filter CMAKE_CUDA_ARCHITECTURES to only those >= a minimum SM version.
# Optionally excludes SM120+ real architectures (for LLM targets that hit
# CCCL tcgen05 PTX issues on Windows/MSVC when compiled for sm_120a native).
#
# Usage:
#   onnxruntime_filter_cuda_archs(<output_var>
#       MIN_SM <number>
#       [EXCLUDE_SM120_REAL])
function(onnxruntime_filter_cuda_archs OUTPUT_VAR)
  cmake_parse_arguments(PARSE_ARGV 1 _FCA "EXCLUDE_SM120_REAL" "MIN_SM" "")

  set(_filtered)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^([0-9]+)" _arch_num "${_arch}")
    if(_arch_num GREATER_EQUAL "${_FCA_MIN_SM}")
      if(_FCA_EXCLUDE_SM120_REAL AND _arch_num GREATER_EQUAL 120 AND _arch MATCHES "-real$")
        continue()
      endif()
      list(APPEND _filtered "${_arch}")
    endif()
  endforeach()
  set("${OUTPUT_VAR}" "${_filtered}" PARENT_SCOPE)
endfunction()

# Create a CUDA OBJECT library for the in-tree CUDA EP and link it to the parent target.
# Uses config_cuda_provider_shared_module() for full configuration (includes, link libs,
# PCH, platform flags, etc.), then applies nvcc --threads.
#
# Usage:
#   onnxruntime_add_cuda_object_library(
#       NAME <target_name>
#       PARENT <parent_target>
#       CUDA_ARCHITECTURES <arch_list>
#       NVCC_THREADS <thread_count>
#       SOURCES <source_files...>)
function(onnxruntime_add_cuda_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES")

  onnxruntime_add_object_library("${_ARG_NAME}" ${_ARG_SOURCES})
  set_target_properties("${_ARG_NAME}" PROPERTIES CUDA_ARCHITECTURES "${_ARG_CUDA_ARCHITECTURES}")
  config_cuda_provider_shared_module("${_ARG_NAME}")
  target_compile_options("${_ARG_NAME}" PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${_ARG_NVCC_THREADS}\">")
  target_link_libraries("${_ARG_PARENT}" PRIVATE "${_ARG_NAME}")
endfunction()

# Create a CUDA OBJECT library for the plugin EP and link it to the parent target.
# Handles the boilerplate: set CUDA_ARCHITECTURES/CUDA_STANDARD, propagate includes
# and compile definitions from the parent, apply shared compile options + nvcc --threads.
#
# Usage:
#   onnxruntime_add_cuda_plugin_object_library(
#       NAME <target_name>
#       PARENT <parent_target>
#       CUDA_ARCHITECTURES <arch_list>
#       NVCC_THREADS <thread_count>
#       COMPILE_OPTIONS <options...>
#       SOURCES <source_files...>)
function(onnxruntime_add_cuda_plugin_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES;COMPILE_OPTIONS")

  onnxruntime_add_object_library("${_ARG_NAME}" ${_ARG_SOURCES})
  set_target_properties("${_ARG_NAME}" PROPERTIES
    CUDA_ARCHITECTURES "${_ARG_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
  target_include_directories("${_ARG_NAME}" PRIVATE
    $<TARGET_PROPERTY:${_ARG_PARENT},INCLUDE_DIRECTORIES>)
  target_compile_definitions("${_ARG_NAME}" PRIVATE
    $<TARGET_PROPERTY:${_ARG_PARENT},COMPILE_DEFINITIONS>)
  target_compile_options("${_ARG_NAME}" PRIVATE
    ${_ARG_COMPILE_OPTIONS}
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${_ARG_NVCC_THREADS}\">")
  target_link_libraries("${_ARG_PARENT}" PRIVATE "${_ARG_NAME}")
endfunction()
