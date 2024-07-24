set(dtype_mapping_void "void")
set(dtype_mapping_fp32 "float")
set(dtype_mapping_fp16 "half")
set(dtype_mapping_bf16 "__nv_bfloat16")
set(dtype_mapping_f8e4m3fn "float8_e4m3fn_t")
set(dtype_mapping_f8e4m3fnuz "float8_e4m3fnuz_t")
set(dtype_mapping_i8 "int8_t")

function(expand_template_to_srcs)
  set(options "")
  set(one_value_keywords "GENERATE" "TEMPLATE" "PREFIX")
  set(multi_value_keywords "DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE")
  cmake_parse_arguments(expand "${options}" "${one_value_keywords}" "${multi_value_keywords}" ${ARGN})

  set(generated_files)
  foreach(item_str ${expand_DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE})
    # list(GET ${item})
    string(REPLACE " " ";" items ${item_str})
    set(items ${items} "")
    list(GET items 0 TIO)
    list(GET items 1 TKV)
    list(GET items 2 TSB)
    list(GET items 3 NUM_THREADS)
    list(GET items 4 HEAD_SIZE)
    list(GET items 5 PAGE_SIZE)
    list(GET items 6 defs)
    set(out "${CMAKE_CURRENT_BINARY_DIR}/${expand_PREFIX}_${TIO}_${TKV}_${TSB}_${NUM_THREADS}_${HEAD_SIZE}_${PAGE_SIZE}.cu")

    set(TIO ${dtype_mapping_${TIO}})
    set(TKV ${dtype_mapping_${TKV}})
    set(TSB ${dtype_mapping_${TSB}})
    configure_file(${expand_TEMPLATE} ${out} @ONLY)

    if (defs)
      string(REPLACE "," ";" defs ${defs})
      set_property(SOURCE ${out} PROPERTY COMPILE_DEFINITIONS ${defs})  # NOTE: will append, overwrite if key is duplicated
    endif()

    list(APPEND generated_files ${out})
  endforeach()

  set_source_files_properties(${generated_files} PROPERTIES GENERATED TRUE)
  set(${expand_GENERATE} ${generated_files} PARENT_SCOPE)
endfunction()

if (NOT paged_attention_template_dir)
  set(paged_attention_template_dir ${CMAKE_CURRENT_LIST_DIR})
endif()

expand_template_to_srcs(
  GENERATE paged_attention_srcs_paged
  TEMPLATE ${paged_attention_template_dir}/paged_attention.cu.in
  PREFIX paged
  DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE
  # fp32 kernels
  "fp32 fp32 void 128 64 8"
  "fp32 fp32 void 128 80 8"
  "fp32 fp32 void 128 96 8"
  "fp32 fp32 void 128 112 8"
  "fp32 fp32 void 128 128 8"
  "fp32 fp32 void 128 256 8"
  "fp32 fp32 void 128 64 16"
  "fp32 fp32 void 128 80 16"
  "fp32 fp32 void 128 96 16"
  "fp32 fp32 void 128 112 16"
  "fp32 fp32 void 128 128 16"
  "fp32 fp32 void 128 256 16"
  "fp32 fp32 void 128 64 32"
  "fp32 fp32 void 128 80 32"
  "fp32 fp32 void 128 96 32"
  "fp32 fp32 void 128 112 32"
  "fp32 fp32 void 128 128 32"
  "fp32 fp32 void 128 256 32"
  # fp16 kernels
  "fp16 fp16 void 128 64 8"
  "fp16 fp16 void 128 80 8"
  "fp16 fp16 void 128 96 8"
  "fp16 fp16 void 128 112 8"
  "fp16 fp16 void 128 128 8"
  "fp16 fp16 void 128 256 8"
  "fp16 fp16 void 128 64 16"
  "fp16 fp16 void 128 80 16"
  "fp16 fp16 void 128 96 16 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16"
  "fp16 fp16 void 128 128 16 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16"
  "fp16 fp16 void 128 64 32"
  "fp16 fp16 void 128 80 32"
  "fp16 fp16 void 128 96 32"
  "fp16 fp16 void 128 112 32"
  "fp16 fp16 void 128 128 32"
  "fp16 fp16 void 128 256 32"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8"
  "fp16 f8e4m3fn fp16 128 80 8"
  "fp16 f8e4m3fn fp16 128 96 8"
  "fp16 f8e4m3fn fp16 128 112 8"
  "fp16 f8e4m3fn fp16 128 128 8"
  "fp16 f8e4m3fn fp16 128 256 8"
  "fp16 f8e4m3fn fp16 128 64 16"
  "fp16 f8e4m3fn fp16 128 80 16"
  "fp16 f8e4m3fn fp16 128 96 16 PAGED_ATTENTION_MAXNREG=80"
  "fp16 f8e4m3fn fp16 128 112 16"
  "fp16 f8e4m3fn fp16 128 128 16 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16"
  "fp16 f8e4m3fn fp16 128 64 32"
  "fp16 f8e4m3fn fp16 128 80 32"
  "fp16 f8e4m3fn fp16 128 96 32"
  "fp16 f8e4m3fn fp16 128 112 32"
  "fp16 f8e4m3fn fp16 128 128 32"
  "fp16 f8e4m3fn fp16 128 256 32"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_ws
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_ws.cu.in
  PREFIX lbp_ws
  DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE
  # fp16 kernels
  "fp16 fp16 void 128 64 8"
  "fp16 fp16 void 128 80 8"
  "fp16 fp16 void 128 96 8"
  "fp16 fp16 void 128 112 8"
  "fp16 fp16 void 128 128 8"
  "fp16 fp16 void 128 256 8"
  "fp16 fp16 void 128 64 16"
  "fp16 fp16 void 128 80 16"
  "fp16 fp16 void 128 96 16"
  "fp16 fp16 void 128 112 16"
  "fp16 fp16 void 128 128 16"
  "fp16 fp16 void 128 256 16"
  "fp16 fp16 void 128 64 32"
  "fp16 fp16 void 128 80 32"
  "fp16 fp16 void 128 96 32"
  "fp16 fp16 void 128 112 32"
  "fp16 fp16 void 128 128 32"
  "fp16 fp16 void 128 256 32"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8"
  "fp16 f8e4m3fn fp16 128 80 8"
  "fp16 f8e4m3fn fp16 128 96 8"
  "fp16 f8e4m3fn fp16 128 112 8"
  "fp16 f8e4m3fn fp16 128 128 8"
  "fp16 f8e4m3fn fp16 128 256 8"
  "fp16 f8e4m3fn fp16 128 64 16"
  "fp16 f8e4m3fn fp16 128 80 16"
  "fp16 f8e4m3fn fp16 128 96 16"
  "fp16 f8e4m3fn fp16 128 112 16"
  "fp16 f8e4m3fn fp16 128 128 16"
  "fp16 f8e4m3fn fp16 128 256 16"
  "fp16 f8e4m3fn fp16 128 64 32"
  "fp16 f8e4m3fn fp16 128 80 32"
  "fp16 f8e4m3fn fp16 128 96 32"
  "fp16 f8e4m3fn fp16 128 112 32"
  "fp16 f8e4m3fn fp16 128 128 32"
  "fp16 f8e4m3fn fp16 128 256 32"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_dpi
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_dpi.cu.in
  PREFIX lbp_dpi
  DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE
  # fp16 kernels
  "fp16 fp16 void 128 64 8"
  "fp16 fp16 void 128 80 8"
  "fp16 fp16 void 128 96 8"
  "fp16 fp16 void 128 112 8"
  "fp16 fp16 void 128 128 8"
  "fp16 fp16 void 128 256 8"
  "fp16 fp16 void 128 64 16"
  "fp16 fp16 void 128 80 16"
  "fp16 fp16 void 128 96 16 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16"
  "fp16 fp16 void 128 128 16 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16"
  "fp16 fp16 void 128 64 32"
  "fp16 fp16 void 128 80 32"
  "fp16 fp16 void 128 96 32"
  "fp16 fp16 void 128 112 32"
  "fp16 fp16 void 128 128 32"
  "fp16 fp16 void 128 256 32"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8"
  "fp16 f8e4m3fn fp16 128 80 8"
  "fp16 f8e4m3fn fp16 128 96 8"
  "fp16 f8e4m3fn fp16 128 112 8"
  "fp16 f8e4m3fn fp16 128 128 8"
  "fp16 f8e4m3fn fp16 128 256 8"
  "fp16 f8e4m3fn fp16 128 64 16"
  "fp16 f8e4m3fn fp16 128 80 16"
  "fp16 f8e4m3fn fp16 128 96 16 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 112 16"
  "fp16 f8e4m3fn fp16 128 128 16 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16"
  "fp16 f8e4m3fn fp16 128 64 32"
  "fp16 f8e4m3fn fp16 128 80 32"
  "fp16 f8e4m3fn fp16 128 96 32"
  "fp16 f8e4m3fn fp16 128 112 32"
  "fp16 f8e4m3fn fp16 128 128 32"
  "fp16 f8e4m3fn fp16 128 256 32"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_dpo
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_dpo.cu.in
  PREFIX lbp_dpo
  DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE
  # fp16 kernels
  "fp16 fp16 void 128 64 8"
  "fp16 fp16 void 128 80 8"
  "fp16 fp16 void 128 96 8"
  "fp16 fp16 void 128 112 8"
  "fp16 fp16 void 128 128 8"
  "fp16 fp16 void 128 256 8"
  "fp16 fp16 void 128 64 16"
  "fp16 fp16 void 128 80 16"
  "fp16 fp16 void 128 96 16 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16"
  "fp16 fp16 void 128 128 16 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16"
  "fp16 fp16 void 128 64 32"
  "fp16 fp16 void 128 80 32"
  "fp16 fp16 void 128 96 32"
  "fp16 fp16 void 128 112 32"
  "fp16 fp16 void 128 128 32"
  "fp16 fp16 void 128 256 32"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8"
  "fp16 f8e4m3fn fp16 128 80 8"
  "fp16 f8e4m3fn fp16 128 96 8"
  "fp16 f8e4m3fn fp16 128 112 8"
  "fp16 f8e4m3fn fp16 128 128 8"
  "fp16 f8e4m3fn fp16 128 256 8"
  "fp16 f8e4m3fn fp16 128 64 16"
  "fp16 f8e4m3fn fp16 128 80 16"
  "fp16 f8e4m3fn fp16 128 96 16 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 112 16"
  "fp16 f8e4m3fn fp16 128 128 16 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16"
  "fp16 f8e4m3fn fp16 128 64 32"
  "fp16 f8e4m3fn fp16 128 80 32"
  "fp16 f8e4m3fn fp16 128 96 32"
  "fp16 f8e4m3fn fp16 128 112 32"
  "fp16 f8e4m3fn fp16 128 128 32"
  "fp16 f8e4m3fn fp16 128 256 32"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_reshape_and_cache
  TEMPLATE ${paged_attention_template_dir}/reshape_and_cache.cu.in
  PREFIX reshape_and_cache
  DTYPEIO_DTYPEKV_DTYPESB_NUMTHREAD_HEADSIZE_PAGESIZE
  "fp16 f8e4m3fn fp16 128 64 8"
  "fp16 f8e4m3fn fp16 128 80 8"
  "fp16 f8e4m3fn fp16 128 96 8"
  "fp16 f8e4m3fn fp16 128 112 8"
  "fp16 f8e4m3fn fp16 128 128 8"
  "fp16 f8e4m3fn fp16 128 256 8"
  "fp16 f8e4m3fn fp16 128 64 16"
  "fp16 f8e4m3fn fp16 128 80 16"
  "fp16 f8e4m3fn fp16 128 96 16"
  "fp16 f8e4m3fn fp16 128 112 16"
  "fp16 f8e4m3fn fp16 128 128 16"
  "fp16 f8e4m3fn fp16 128 256 16"
  "fp16 f8e4m3fn fp16 128 64 32"
  "fp16 f8e4m3fn fp16 128 80 32"
  "fp16 f8e4m3fn fp16 128 96 32"
  "fp16 f8e4m3fn fp16 128 112 32"
  "fp16 f8e4m3fn fp16 128 128 32"
  "fp16 f8e4m3fn fp16 128 256 32"
)

set(paged_attention_generated_srcs
  ${paged_attention_srcs_paged}
  ${paged_attention_srcs_lbp_ws}
  ${paged_attention_srcs_lbp_dpi}
  ${paged_attention_srcs_lbp_dpo}
  ${paged_attention_srcs_reshape_and_cache}
)
