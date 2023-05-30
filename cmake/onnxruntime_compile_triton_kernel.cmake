# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# set all triton kernel ops that need to be compiled
set(triton_kernel_scripts
    "onnxruntime/core/providers/rocm/math/softmax_triton.py"
    "onnxruntime/contrib_ops/rocm/diffusion/group_norm_triton.py"
)

function(compile_triton_kernel out_triton_kernel_obj_file out_triton_kernel_header_dir)
  # compile triton kernel, generate .a and .h files
  set(triton_kernel_compiler "${REPO_ROOT}/tools/ci_build/compile_triton.py")
  set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/triton_kernels")
  set(out_obj_file "${out_dir}/triton_kernel_infos.a")
  set(header_file "${out_dir}/triton_kernel_infos.h")

  list(TRANSFORM triton_kernel_scripts PREPEND "${REPO_ROOT}/")

  add_custom_command(
    OUTPUT ${out_obj_file} ${header_file}
    COMMAND Python3::Interpreter ${triton_kernel_compiler}
            --header ${header_file}
            --script_files ${triton_kernel_scripts}
            --obj_file ${out_obj_file}
    DEPENDS ${triton_kernel_scripts} ${triton_kernel_compiler}
    COMMENT "Triton compile generates: ${out_obj_file}"
  )
  add_custom_target(onnxruntime_triton_kernel DEPENDS ${out_obj_file} ${header_file})
  set(${out_triton_kernel_obj_file} ${out_obj_file} PARENT_SCOPE)
  set(${out_triton_kernel_header_dir} ${out_dir} PARENT_SCOPE)
endfunction()
