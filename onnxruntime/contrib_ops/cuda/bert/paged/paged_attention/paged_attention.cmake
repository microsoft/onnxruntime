if (NOT paged_attention_template_dir)
  set(paged_attention_template_dir ${CMAKE_CURRENT_LIST_DIR})
endif()

set(dtype_mapping_void "void")
set(dtype_mapping_fp32 "float")
set(dtype_mapping_fp16 "half")
set(dtype_mapping_bf16 "__nv_bfloat16")
set(dtype_mapping_f8e4m3fn "float8_e4m3fn_t")
set(dtype_mapping_f8e4m3fnuz "float8_e4m3fnuz_t")
set(dtype_mapping_i8 "int8_t")

function(expand_template_to_srcs)
  set(options "")
  set(one_value_keywords "GENERATE" "TEMPLATE" "PREFIX" "NUM_QUERIES_PER_KV")
  set(multi_value_keywords "CONFIGS")
  cmake_parse_arguments(expand "${options}" "${one_value_keywords}" "${multi_value_keywords}" ${ARGN})

  list(GET expand_CONFIGS 0 config_header)
  list(REMOVE_AT expand_CONFIGS 0)
  string(REPLACE " " ";" config_header ${config_header})
  list(LENGTH config_header num_keywords)  # then value at num_keywords is defs
  math(EXPR num_keywords_1 "${num_keywords}-1")

  set(generated_files)
  foreach(item_str ${expand_CONFIGS})
    # list(GET ${item})
    string(REPLACE " " ";" items ${item_str})
    set(items ${items} "")
    set(basename)
    foreach(idx RANGE ${num_keywords_1})
      list(GET config_header ${idx} key)
      list(GET items ${idx} value)
      string(APPEND basename " ${value}")

      # apply local value for local expend
      set(dtype_expended "${dtype_mapping_${value}}")
      if (dtype_expended)
        set(${key} ${dtype_expended})
      else()
        set(${key} ${value})
      endif()
    endforeach()

    list(GET items ${num_keywords} defs)  # then value at num_keywords is defs
    string(STRIP ${basename} basename)
    string(REPLACE " " "_" basename ${basename})

    set(out "${CMAKE_CURRENT_BINARY_DIR}/${expand_PREFIX}_${basename}.cu")
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

expand_template_to_srcs(
  GENERATE paged_attention_srcs_paged
  TEMPLATE ${paged_attention_template_dir}/paged_attention.cu.in
  PREFIX paged
  CONFIGS
  "TIO TKV TSB NUM_THREADS HEAD_SIZE PAGE_SIZE NUM_QUERIES_PER_KV"
  # fp32 kernels
  "fp32 fp32 void 128 64 8 1"
  "fp32 fp32 void 128 80 8 1"
  "fp32 fp32 void 128 96 8 1"
  "fp32 fp32 void 128 112 8 1"
  "fp32 fp32 void 128 128 8 1"
  "fp32 fp32 void 128 256 8 1"
  "fp32 fp32 void 128 64 16 1"
  "fp32 fp32 void 128 80 16 1"
  "fp32 fp32 void 128 96 16 1"
  "fp32 fp32 void 128 112 16 1"
  "fp32 fp32 void 128 128 16 1"
  "fp32 fp32 void 128 256 16 1"
  "fp32 fp32 void 128 64 32 1"
  "fp32 fp32 void 128 80 32 1"
  "fp32 fp32 void 128 96 32 1"
  "fp32 fp32 void 128 112 32 1"
  "fp32 fp32 void 128 128 32 1"
  "fp32 fp32 void 128 256 32 1"
  # fp16 kernels
  "fp16 fp16 void 128 64 8 1"
  "fp16 fp16 void 128 80 8 1"
  "fp16 fp16 void 128 96 8 1"
  "fp16 fp16 void 128 112 8 1"
  "fp16 fp16 void 128 128 8 1"
  "fp16 fp16 void 128 256 8 1"
  "fp16 fp16 void 128 64 16 1"
  "fp16 fp16 void 128 80 16 1"
  "fp16 fp16 void 128 96 16 1 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16 1"
  "fp16 fp16 void 128 128 16 1 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16 1"
  "fp16 fp16 void 128 64 32 1"
  "fp16 fp16 void 128 80 32 1"
  "fp16 fp16 void 128 96 32 1"
  "fp16 fp16 void 128 112 32 1"
  "fp16 fp16 void 128 128 32 1"
  "fp16 fp16 void 128 256 32 1"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8 1"
  "fp16 f8e4m3fn fp16 128 80 8 1"
  "fp16 f8e4m3fn fp16 128 96 8 1"
  "fp16 f8e4m3fn fp16 128 112 8 1"
  "fp16 f8e4m3fn fp16 128 128 8 1"
  "fp16 f8e4m3fn fp16 128 256 8 1"
  "fp16 f8e4m3fn fp16 128 64 16 1"
  "fp16 f8e4m3fn fp16 128 80 16 1"
  "fp16 f8e4m3fn fp16 128 96 16 1 PAGED_ATTENTION_MAXNREG=80"
  "fp16 f8e4m3fn fp16 128 112 16 1"
  "fp16 f8e4m3fn fp16 128 128 16 1 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16 1"
  "fp16 f8e4m3fn fp16 128 64 32 1"
  "fp16 f8e4m3fn fp16 128 80 32 1"
  "fp16 f8e4m3fn fp16 128 96 32 1"
  "fp16 f8e4m3fn fp16 128 112 32 1"
  "fp16 f8e4m3fn fp16 128 128 32 1"
  "fp16 f8e4m3fn fp16 128 256 32 1"
  # fp32 kernels gqa4
  "fp32 fp32 void 128 64 8 4"
  "fp32 fp32 void 128 80 8 4"
  "fp32 fp32 void 128 96 8 4"
  "fp32 fp32 void 128 112 8 4"
  "fp32 fp32 void 128 128 8 4"
  "fp32 fp32 void 128 256 8 4"
  "fp32 fp32 void 128 64 16 4"
  "fp32 fp32 void 128 80 16 4"
  "fp32 fp32 void 128 96 16 4"
  "fp32 fp32 void 128 112 16 4"
  "fp32 fp32 void 128 128 16 4"
  "fp32 fp32 void 128 256 16 4"
  "fp32 fp32 void 128 64 32 4"
  "fp32 fp32 void 128 80 32 4"
  "fp32 fp32 void 128 96 32 4"
  "fp32 fp32 void 128 112 32 4"
  "fp32 fp32 void 128 128 32 4"
  "fp32 fp32 void 128 256 32 4"
  # fp16 kernels gqa4
  "fp16 fp16 void 128 64 8 4"
  "fp16 fp16 void 128 80 8 4"
  "fp16 fp16 void 128 96 8 4"
  "fp16 fp16 void 128 112 8 4"
  "fp16 fp16 void 128 128 8 4"
  "fp16 fp16 void 128 256 8 4"
  "fp16 fp16 void 128 64 16 4"
  "fp16 fp16 void 128 80 16 4"
  "fp16 fp16 void 128 96 16 4"
  "fp16 fp16 void 128 112 16 4"
  "fp16 fp16 void 128 128 16 4"
  "fp16 fp16 void 128 256 16 4"
  "fp16 fp16 void 128 64 32 4"
  "fp16 fp16 void 128 80 32 4"
  "fp16 fp16 void 128 96 32 4"
  "fp16 fp16 void 128 112 32 4"
  "fp16 fp16 void 128 128 32 4"
  "fp16 fp16 void 128 256 32 4"
  # fp8 kernels gqa4
  "fp16 f8e4m3fn fp16 128 64 8 4"
  "fp16 f8e4m3fn fp16 128 80 8 4"
  "fp16 f8e4m3fn fp16 128 96 8 4"
  "fp16 f8e4m3fn fp16 128 112 8 4"
  "fp16 f8e4m3fn fp16 128 128 8 4"
  "fp16 f8e4m3fn fp16 128 256 8 4"
  "fp16 f8e4m3fn fp16 128 64 16 4"
  "fp16 f8e4m3fn fp16 128 80 16 4"
  "fp16 f8e4m3fn fp16 128 96 16 4"
  "fp16 f8e4m3fn fp16 128 112 16 4"
  "fp16 f8e4m3fn fp16 128 128 16 4"
  "fp16 f8e4m3fn fp16 128 256 16 4"
  "fp16 f8e4m3fn fp16 128 64 32 4"
  "fp16 f8e4m3fn fp16 128 80 32 4"
  "fp16 f8e4m3fn fp16 128 96 32 4"
  "fp16 f8e4m3fn fp16 128 112 32 4"
  "fp16 f8e4m3fn fp16 128 128 32 4"
  "fp16 f8e4m3fn fp16 128 256 32 4"
  # fp16 kernels gqa8
  "fp16 fp16 void 128 64 8 8"
  "fp16 fp16 void 128 80 8 8"
  "fp16 fp16 void 128 96 8 8"
  "fp16 fp16 void 128 112 8 8"
  "fp16 fp16 void 128 128 8 8"
  "fp16 fp16 void 128 256 8 8"
  "fp16 fp16 void 128 64 16 8"
  "fp16 fp16 void 128 80 16 8"
  "fp16 fp16 void 128 96 16 8"
  "fp16 fp16 void 128 112 16 8"
  "fp16 fp16 void 128 128 16 8"
  "fp16 fp16 void 128 256 16 8"
  "fp16 fp16 void 128 64 32 8"
  "fp16 fp16 void 128 80 32 8"
  "fp16 fp16 void 128 96 32 8"
  "fp16 fp16 void 128 112 32 8"
  "fp16 fp16 void 128 128 32 8"
  "fp16 fp16 void 128 256 32 8"
  # fp8 kernels gqa8
  "fp16 f8e4m3fn fp16 128 64 8 8"
  "fp16 f8e4m3fn fp16 128 80 8 8"
  "fp16 f8e4m3fn fp16 128 96 8 8"
  "fp16 f8e4m3fn fp16 128 112 8 8"
  "fp16 f8e4m3fn fp16 128 128 8 8"
  "fp16 f8e4m3fn fp16 128 256 8 8"
  "fp16 f8e4m3fn fp16 128 64 16 8"
  "fp16 f8e4m3fn fp16 128 80 16 8"
  "fp16 f8e4m3fn fp16 128 96 16 8"
  "fp16 f8e4m3fn fp16 128 112 16 8"
  "fp16 f8e4m3fn fp16 128 128 16 8"
  "fp16 f8e4m3fn fp16 128 256 16 8"
  "fp16 f8e4m3fn fp16 128 64 32 8"
  "fp16 f8e4m3fn fp16 128 80 32 8"
  "fp16 f8e4m3fn fp16 128 96 32 8"
  "fp16 f8e4m3fn fp16 128 112 32 8"
  "fp16 f8e4m3fn fp16 128 128 32 8"
  "fp16 f8e4m3fn fp16 128 256 32 8"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_ws
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_ws.cu.in
  PREFIX lbp_ws
  CONFIGS
  "TIO TKV TSB NUM_THREADS HEAD_SIZE PAGE_SIZE"
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
  CONFIGS
  "TIO TKV TSB NUM_THREADS HEAD_SIZE PAGE_SIZE NUM_QUERIES_PER_KV"
  # fp16 kernels
  "fp16 fp16 void 128 64 8 1"
  "fp16 fp16 void 128 80 8 1"
  "fp16 fp16 void 128 96 8 1"
  "fp16 fp16 void 128 112 8 1"
  "fp16 fp16 void 128 128 8 1"
  "fp16 fp16 void 128 256 8 1"
  "fp16 fp16 void 128 64 16 1"
  "fp16 fp16 void 128 80 16 1"
  "fp16 fp16 void 128 96 16 1 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16 1"
  "fp16 fp16 void 128 128 16 1 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16 1"
  "fp16 fp16 void 128 64 32 1"
  "fp16 fp16 void 128 80 32 1"
  "fp16 fp16 void 128 96 32 1"
  "fp16 fp16 void 128 112 32 1"
  "fp16 fp16 void 128 128 32 1"
  "fp16 fp16 void 128 256 32 1"
  # fp8 kernels
  "fp16 f8e4m3fn fp16 128 64 8 1"
  "fp16 f8e4m3fn fp16 128 80 8 1"
  "fp16 f8e4m3fn fp16 128 96 8 1"
  "fp16 f8e4m3fn fp16 128 112 8 1"
  "fp16 f8e4m3fn fp16 128 128 8 1"
  "fp16 f8e4m3fn fp16 128 256 8 1"
  "fp16 f8e4m3fn fp16 128 64 16 1"
  "fp16 f8e4m3fn fp16 128 80 16 1"
  "fp16 f8e4m3fn fp16 128 96 16 1 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 112 16 1"
  "fp16 f8e4m3fn fp16 128 128 16 1 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16 1"
  "fp16 f8e4m3fn fp16 128 64 32 1"
  "fp16 f8e4m3fn fp16 128 80 32 1"
  "fp16 f8e4m3fn fp16 128 96 32 1"
  "fp16 f8e4m3fn fp16 128 112 32 1"
  "fp16 f8e4m3fn fp16 128 128 32 1"
  "fp16 f8e4m3fn fp16 128 256 32 1"
  # fp16 kernels gqa4
  "fp16 fp16 void 128 64 8 4"
  "fp16 fp16 void 128 80 8 4"
  "fp16 fp16 void 128 96 8 4"
  "fp16 fp16 void 128 112 8 4"
  "fp16 fp16 void 128 128 8 4"
  "fp16 fp16 void 128 256 8 4"
  "fp16 fp16 void 128 64 16 4"
  "fp16 fp16 void 128 80 16 4"
  "fp16 fp16 void 128 96 16 4"
  "fp16 fp16 void 128 112 16 4"
  "fp16 fp16 void 128 128 16 4"
  "fp16 fp16 void 128 256 16 4"
  "fp16 fp16 void 128 64 32 4"
  "fp16 fp16 void 128 80 32 4"
  "fp16 fp16 void 128 96 32 4"
  "fp16 fp16 void 128 112 32 4"
  "fp16 fp16 void 128 128 32 4"
  "fp16 fp16 void 128 256 32 4"
  # fp8 kernels gqa4
  "fp16 f8e4m3fn fp16 128 64 8 4"
  "fp16 f8e4m3fn fp16 128 80 8 4"
  "fp16 f8e4m3fn fp16 128 96 8 4"
  "fp16 f8e4m3fn fp16 128 112 8 4"
  "fp16 f8e4m3fn fp16 128 128 8 4"
  "fp16 f8e4m3fn fp16 128 256 8 4"
  "fp16 f8e4m3fn fp16 128 64 16 4"
  "fp16 f8e4m3fn fp16 128 80 16 4"
  "fp16 f8e4m3fn fp16 128 96 16 4"
  "fp16 f8e4m3fn fp16 128 112 16 4"
  "fp16 f8e4m3fn fp16 128 128 16 4"
  "fp16 f8e4m3fn fp16 128 256 16 4"
  "fp16 f8e4m3fn fp16 128 64 32 4"
  "fp16 f8e4m3fn fp16 128 80 32 4"
  "fp16 f8e4m3fn fp16 128 96 32 4"
  "fp16 f8e4m3fn fp16 128 112 32 4"
  "fp16 f8e4m3fn fp16 128 128 32 4"
  "fp16 f8e4m3fn fp16 128 256 32 4"
  # fp16 kernels gqa8
  "fp16 fp16 void 128 64 8 8"
  "fp16 fp16 void 128 80 8 8"
  "fp16 fp16 void 128 96 8 8"
  "fp16 fp16 void 128 112 8 8"
  "fp16 fp16 void 128 128 8 8"
  "fp16 fp16 void 128 256 8 8"
  "fp16 fp16 void 128 64 16 8"
  "fp16 fp16 void 128 80 16 8"
  "fp16 fp16 void 128 96 16 8"
  "fp16 fp16 void 128 112 16 8"
  "fp16 fp16 void 128 128 16 8"
  "fp16 fp16 void 128 256 16 8"
  "fp16 fp16 void 128 64 32 8"
  "fp16 fp16 void 128 80 32 8"
  "fp16 fp16 void 128 96 32 8"
  "fp16 fp16 void 128 112 32 8"
  "fp16 fp16 void 128 128 32 8"
  "fp16 fp16 void 128 256 32 8"
  # fp8 kernels gqa8
  "fp16 f8e4m3fn fp16 128 64 8 8"
  "fp16 f8e4m3fn fp16 128 80 8 8"
  "fp16 f8e4m3fn fp16 128 96 8 8"
  "fp16 f8e4m3fn fp16 128 112 8 8"
  "fp16 f8e4m3fn fp16 128 128 8 8"
  "fp16 f8e4m3fn fp16 128 256 8 8"
  "fp16 f8e4m3fn fp16 128 64 16 8"
  "fp16 f8e4m3fn fp16 128 80 16 8"
  "fp16 f8e4m3fn fp16 128 96 16 8"
  "fp16 f8e4m3fn fp16 128 112 16 8"
  "fp16 f8e4m3fn fp16 128 128 16 8"
  "fp16 f8e4m3fn fp16 128 256 16 8"
  "fp16 f8e4m3fn fp16 128 64 32 8"
  "fp16 f8e4m3fn fp16 128 80 32 8"
  "fp16 f8e4m3fn fp16 128 96 32 8"
  "fp16 f8e4m3fn fp16 128 112 32 8"
  "fp16 f8e4m3fn fp16 128 128 32 8"
  "fp16 f8e4m3fn fp16 128 256 32 8"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_dpo
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_dpo.cu.in
  PREFIX lbp_dpo
  CONFIGS
  "TIO TKV TSB NUM_THREADS HEAD_SIZE PAGE_SIZE NUM_QUERIES_PER_KV"
  # fp16 kernels mha
  "fp16 fp16 void 128 64 8 1"
  "fp16 fp16 void 128 80 8 1"
  "fp16 fp16 void 128 96 8 1"
  "fp16 fp16 void 128 112 8 1"
  "fp16 fp16 void 128 128 8 1"
  "fp16 fp16 void 128 256 8 1"
  "fp16 fp16 void 128 64 16 1"
  "fp16 fp16 void 128 80 16 1"
  "fp16 fp16 void 128 96 16 1 PAGED_ATTENTION_MAXNREG=72"
  "fp16 fp16 void 128 112 16 1"
  "fp16 fp16 void 128 128 16 1 PAGED_ATTENTION_MAXNREG=80"
  "fp16 fp16 void 128 256 16 1"
  "fp16 fp16 void 128 64 32 1"
  "fp16 fp16 void 128 80 32 1"
  "fp16 fp16 void 128 96 32 1"
  "fp16 fp16 void 128 112 32 1"
  "fp16 fp16 void 128 128 32 1"
  "fp16 fp16 void 128 256 32 1"
  # fp8 kernels mha
  "fp16 f8e4m3fn fp16 128 64 8 1"
  "fp16 f8e4m3fn fp16 128 80 8 1"
  "fp16 f8e4m3fn fp16 128 96 8 1"
  "fp16 f8e4m3fn fp16 128 112 8 1"
  "fp16 f8e4m3fn fp16 128 128 8 1"
  "fp16 f8e4m3fn fp16 128 256 8 1"
  "fp16 f8e4m3fn fp16 128 64 16 1"
  "fp16 f8e4m3fn fp16 128 80 16 1"
  "fp16 f8e4m3fn fp16 128 96 16 1 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 112 16 1"
  "fp16 f8e4m3fn fp16 128 128 16 1 PAGED_ATTENTION_MAXNREG=96"
  "fp16 f8e4m3fn fp16 128 256 16 1"
  "fp16 f8e4m3fn fp16 128 64 32 1"
  "fp16 f8e4m3fn fp16 128 80 32 1"
  "fp16 f8e4m3fn fp16 128 96 32 1"
  "fp16 f8e4m3fn fp16 128 112 32 1"
  "fp16 f8e4m3fn fp16 128 128 32 1"
  "fp16 f8e4m3fn fp16 128 256 32 1"
  # fp16 kernels gqa4
  "fp16 fp16 void 128 64 8 4"
  "fp16 fp16 void 128 80 8 4"
  "fp16 fp16 void 128 96 8 4"
  "fp16 fp16 void 128 112 8 4"
  "fp16 fp16 void 128 128 8 4"
  "fp16 fp16 void 128 256 8 4"
  "fp16 fp16 void 128 64 16 4"
  "fp16 fp16 void 128 80 16 4"
  "fp16 fp16 void 128 96 16 4"
  "fp16 fp16 void 128 112 16 4"
  "fp16 fp16 void 128 128 16 4"
  "fp16 fp16 void 128 256 16 4"
  "fp16 fp16 void 128 64 32 4"
  "fp16 fp16 void 128 80 32 4"
  "fp16 fp16 void 128 96 32 4"
  "fp16 fp16 void 128 112 32 4"
  "fp16 fp16 void 128 128 32 4"
  "fp16 fp16 void 128 256 32 4"
  # fp8 kernels gqa4
  "fp16 f8e4m3fn fp16 128 64 8 4"
  "fp16 f8e4m3fn fp16 128 80 8 4"
  "fp16 f8e4m3fn fp16 128 96 8 4"
  "fp16 f8e4m3fn fp16 128 112 8 4"
  "fp16 f8e4m3fn fp16 128 128 8 4"
  "fp16 f8e4m3fn fp16 128 256 8 4"
  "fp16 f8e4m3fn fp16 128 64 16 4"
  "fp16 f8e4m3fn fp16 128 80 16 4"
  "fp16 f8e4m3fn fp16 128 96 16 4"
  "fp16 f8e4m3fn fp16 128 112 16 4"
  "fp16 f8e4m3fn fp16 128 128 16 4"
  "fp16 f8e4m3fn fp16 128 256 16 4"
  "fp16 f8e4m3fn fp16 128 64 32 4"
  "fp16 f8e4m3fn fp16 128 80 32 4"
  "fp16 f8e4m3fn fp16 128 96 32 4"
  "fp16 f8e4m3fn fp16 128 112 32 4"
  "fp16 f8e4m3fn fp16 128 128 32 4"
  "fp16 f8e4m3fn fp16 128 256 32 4"
  # fp16 kernels gqa8
  "fp16 fp16 void 128 64 8 8"
  "fp16 fp16 void 128 80 8 8"
  "fp16 fp16 void 128 96 8 8"
  "fp16 fp16 void 128 112 8 8"
  "fp16 fp16 void 128 128 8 8"
  "fp16 fp16 void 128 256 8 8"
  "fp16 fp16 void 128 64 16 8"
  "fp16 fp16 void 128 80 16 8"
  "fp16 fp16 void 128 96 16 8"
  "fp16 fp16 void 128 112 16 8"
  "fp16 fp16 void 128 128 16 8"
  "fp16 fp16 void 128 256 16 8"
  "fp16 fp16 void 128 64 32 8"
  "fp16 fp16 void 128 80 32 8"
  "fp16 fp16 void 128 96 32 8"
  "fp16 fp16 void 128 112 32 8"
  "fp16 fp16 void 128 128 32 8"
  "fp16 fp16 void 128 256 32 8"
  # fp8 kernels gqa8
  "fp16 f8e4m3fn fp16 128 64 8 8"
  "fp16 f8e4m3fn fp16 128 80 8 8"
  "fp16 f8e4m3fn fp16 128 96 8 8"
  "fp16 f8e4m3fn fp16 128 112 8 8"
  "fp16 f8e4m3fn fp16 128 128 8 8"
  "fp16 f8e4m3fn fp16 128 256 8 8"
  "fp16 f8e4m3fn fp16 128 64 16 8"
  "fp16 f8e4m3fn fp16 128 80 16 8"
  "fp16 f8e4m3fn fp16 128 96 16 8"
  "fp16 f8e4m3fn fp16 128 112 16 8"
  "fp16 f8e4m3fn fp16 128 128 16 8"
  "fp16 f8e4m3fn fp16 128 256 16 8"
  "fp16 f8e4m3fn fp16 128 64 32 8"
  "fp16 f8e4m3fn fp16 128 80 32 8"
  "fp16 f8e4m3fn fp16 128 96 32 8"
  "fp16 f8e4m3fn fp16 128 112 32 8"
  "fp16 f8e4m3fn fp16 128 128 32 8"
  "fp16 f8e4m3fn fp16 128 256 32 8"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_lbp_dpo_reduction
  TEMPLATE ${paged_attention_template_dir}/lbp_attention_dpo_reduction.cu.in
  PREFIX lbp_dpo_reduction
  CONFIGS
  "TIO NUM_THREADS HEAD_SIZE"
  # fp32 kernels
  "fp32 128 64 INSTANTIATE_SPLIT_HEAD_KERNEL=1"
  "fp32 128 80"
  "fp32 128 96"
  "fp32 128 112"
  "fp32 128 128"
  "fp32 128 256"
  # fp16 kernels
  "fp16 128 64 INSTANTIATE_SPLIT_HEAD_KERNEL=1"
  "fp16 128 80"
  "fp16 128 96"
  "fp16 128 112"
  "fp16 128 128"
  "fp16 128 256"
)

expand_template_to_srcs(
  GENERATE paged_attention_srcs_reshape_and_cache
  TEMPLATE ${paged_attention_template_dir}/reshape_and_cache.cu.in
  PREFIX reshape_and_cache
  CONFIGS
  "TIO TKV TSB NUM_THREADS HEAD_SIZE PAGE_SIZE"
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
  ${paged_attention_srcs_lbp_dpo_reduction}
  ${paged_attention_srcs_reshape_and_cache}
)
