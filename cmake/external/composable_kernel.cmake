set(PATCH ${PROJECT_SOURCE_DIR}/patches/composable_kernel/Fix_Clang_Build.patch)

include(FetchContent)
onnxruntime_fetchcontent_declare(composable_kernel
  URL ${DEP_URL_composable_kernel}
  URL_HASH SHA1=${DEP_SHA1_composable_kernel}
  PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PATCH}
  EXCLUDE_FROM_ALL
)

FetchContent_GetProperties(composable_kernel)
if(NOT composable_kernel_POPULATED)
  FetchContent_Populate(composable_kernel)
  set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  set(BUILD_DEV OFF CACHE BOOL "Disable -Weverything, otherwise, error: 'constexpr' specifier is incompatible with C++98 [-Werror,-Wc++98-compat]" FORCE)
  # Exclude i8 device gemm instances due to excessive long compilation time and not being used
  set(DTYPES fp32 fp16 bf16 fp8)
  set(INSTANCES_ONLY ON)
  add_subdirectory(${composable_kernel_SOURCE_DIR} ${composable_kernel_BINARY_DIR} EXCLUDE_FROM_ALL)

  add_library(onnxruntime_composable_kernel_includes INTERFACE)
  target_include_directories(onnxruntime_composable_kernel_includes INTERFACE
    ${composable_kernel_SOURCE_DIR}/include
    ${composable_kernel_BINARY_DIR}/include
    ${composable_kernel_SOURCE_DIR}/library/include)
  target_compile_definitions(onnxruntime_composable_kernel_includes INTERFACE __fp32__ __fp16__ __bf16__)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py
    --list_blobs ${composable_kernel_BINARY_DIR}/blob_list.txt
    COMMAND_ERROR_IS_FATAL ANY
  )
  file(STRINGS ${composable_kernel_BINARY_DIR}/blob_list.txt generated_fmha_srcs)
  add_custom_command(
    OUTPUT ${generated_fmha_srcs}
    COMMAND ${Python3_EXECUTABLE} ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py --output_dir ${composable_kernel_BINARY_DIR}
    DEPENDS ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py ${composable_kernel_BINARY_DIR}/blob_list.txt
  )
  set_source_files_properties(${generated_fmha_srcs} PROPERTIES LANGUAGE HIP GENERATED TRUE)
  add_custom_target(gen_fmha_srcs DEPENDS ${generated_fmha_srcs})  # dummy target for dependencies
  # code generation complete

  set(fmha_srcs
    ${generated_fmha_srcs}
    ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/fmha_fwd.cpp
    ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/fmha_fwd.hpp
    ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/bias.hpp
    ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha/mask.hpp
  )
  add_library(onnxruntime_composable_kernel_fmha STATIC EXCLUDE_FROM_ALL ${generated_fmha_srcs})
  target_link_libraries(onnxruntime_composable_kernel_fmha PUBLIC onnxruntime_composable_kernel_includes)
  target_include_directories(onnxruntime_composable_kernel_fmha PUBLIC ${composable_kernel_SOURCE_DIR}/example/ck_tile/01_fmha)
  add_dependencies(onnxruntime_composable_kernel_fmha gen_fmha_srcs)

  # ck tile only supports MI200+ GPUs at the moment
  get_target_property(archs onnxruntime_composable_kernel_fmha HIP_ARCHITECTURES)
  string(REPLACE "," ";" archs "${archs}")
  set(original_archs ${archs})
  list(FILTER archs INCLUDE REGEX "(gfx942|gfx90a)")
  if (NOT original_archs EQUAL archs)
    message(WARNING "ck tile only supports archs: ${archs} among the originally specified ${original_archs}")
  endif()
  set_target_properties(onnxruntime_composable_kernel_fmha PROPERTIES HIP_ARCHITECTURES "${archs}")
endif()
