# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_providers_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc"
)

file(GLOB_RECURSE onnxruntime_cpu_contrib_ops_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu_contrib_kernels.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu_contrib_kernels.cc"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc"
)

file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda_contrib_kernels.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda_contrib_kernels.cc"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cc"
)

file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cu_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cu"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cuh"
)

file(GLOB onnxruntime_cpu_automl_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/automl_ops/cpu_automl_kernels.h"
  "${ONNXRUNTIME_ROOT}/automl_ops/cpu_automl_kernels.cc"
  "${ONNXRUNTIME_ROOT}/automl_ops/automl_types.h"
  "${ONNXRUNTIME_ROOT}/automl_ops/automl_types.cc"
  "${ONNXRUNTIME_ROOT}/automl_ops/automl_featurizers.h"
  "${ONNXRUNTIME_ROOT}/automl_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/automl_ops/cpu/*.cc"
)

file(GLOB onnxruntime_providers_common_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/*.cc"
)

if(onnxruntime_USE_MKLDNN)
  set(PROVIDERS_MKLDNN onnxruntime_providers_mkldnn)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES mkldnn)
endif()
if(onnxruntime_USE_NGRAPH)
  set(PROVIDERS_NGRAPH onnxruntime_providers_ngraph)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES ngraph)
endif()
if(onnxruntime_USE_NUPHAR)
  set(PROVIDERS_NUPHAR onnxruntime_providers_nuphar)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES nuphar)
endif()
if(onnxruntime_USE_CUDA)
  set(PROVIDERS_CUDA onnxruntime_providers_cuda)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)
endif()
if(onnxruntime_USE_TENSORRT)
  set(PROVIDERS_TENSORRT onnxruntime_providers_tensorrt)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES tensorrt)
endif()
if(onnxruntime_USE_OPENVINO)
  set(PROVIDERS_OPENVINO onnxruntime_providers_openvino)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES openvino)
endif()
if(onnxruntime_USE_NNAPI)
  set(PROVIDERS_NNAPI onnxruntime_providers_nnapi)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES nnapi)
endif()
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

set(onnxruntime_providers_src ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

# disable contrib ops conditionally
if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cpu_contrib_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_contrib_ops_srcs})
endif()

if (onnxruntime_USE_AUTOML)
  source_group(TREE ${ONNXRUNTIME_ROOT}/ FILES ${onnxruntime_cpu_automl_cc_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_automl_cc_srcs})
endif()

add_library(onnxruntime_providers ${onnxruntime_providers_src})
onnxruntime_add_include_to_target(onnxruntime_providers onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)

if (onnxruntime_USE_AUTOML)
  add_dependencies(onnxruntime_providers automl_featurizers)
  onnxruntime_add_include_to_target(onnxruntime_providers automl_featurizers)
  target_link_libraries(onnxruntime_providers automl_featurizers)
endif()

if(HAS_DEPRECATED_COPY)
  #temporarily ignore this warning
  #see: https://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/matmul_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/quantize_linear_matmul.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/qlinearconv.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/conv_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/generator/random.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/onehot.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/where_op.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
endif()

set(re2_src ${ONNXRUNTIME_ROOT}/../cmake/external/re2)
target_include_directories(onnxruntime_providers PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${gemmlowp_src} ${re2_src})
add_dependencies(onnxruntime_providers onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "ONNXRuntime")


if (onnxruntime_USE_MIMALLOC)
  set(mimalloc_root_dir ${PROJECT_SOURCE_DIR}/external/mimalloc)
  set(mimalloc_output_dir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/)
  set(mimalloc_wheel_dir ${mimalloc_output_dir}/onnxruntime/capi/)

  add_definitions(
    -DUSE_MIMALLOC=1 # used in ONNXRuntime
    -DMI_OVERRIDE=ON) # used in building MiMalloc
  include_directories(${mimalloc_root_dir}/include)
  
  if(NOT IS_DIRECTORY ${mimalloc_wheel_dir})
    file(MAKE_DIRECTORY ${mimalloc_wheel_dir})
  endif()
  
  if (WIN32)
    # The generic MiMalloc CMakeLists.txt project lacks 
    # the needed hooks to override malloc at runtime on Windows
    # so we fall back to the specially provided VS solutions (which
    # do have those hooks)
    set(mimalloc_output mimalloc-override)

    if(NOT ${CMAKE_GENERATOR_PLATFORM} MATCHES "x64|Win32")
      message(FATAL_ERROR "MiMalloc doesn't support ARM/ARM64 targets")
    endif()
    
    set(vs_version "vs2019")
    if (${CMAKE_GENERATOR} MATCHES "Visual Studio [1-5]+ [0-9]+")
      set(vs_version "vs2017")
    endif()
    
    set(mimalloc_config "Release")
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
      set(mimalloc_config, "Debug")
    endif()

    set(mimalloc_target_winsdk ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION})
    if(DEFINED ENV{WindowsSDKVersion})
      set(mimalloc_target_winsdk $ENV{WindowsSDKVersion})
    endif()
    
    set(mimalloc_deps ${mimalloc_output_dir}mimalloc-redirect.dll)
    set(mimalloc_platform ${CMAKE_GENERATOR_PLATFORM})
    if(${CMAKE_GENERATOR_PLATFORM} MATCHES "Win32")
      set(mimalloc_deps ${mimalloc_output_dir}mimalloc-redirect32.dll)
      set(mimalloc_platform x86)
    endif()

    # msbuild throws a fit during a postbuild step when copying files if the source uses backslashes and the destination uses forward slashes
    STRING(REGEX REPLACE "/" "\\\\" msbuild_converted_output_dir ${mimalloc_output_dir})
    add_custom_command(OUTPUT ${mimalloc_output} COMMAND msbuild ${mimalloc_root_dir}/ide/${vs_version}/mimalloc.sln
                      /p:OutDir=${msbuild_converted_output_dir} /p:Platform=${mimalloc_platform} /p:Configuration=${mimalloc_config}
                      /p:WindowsTargetPlatformVersion=${mimalloc_target_winsdk})
    add_custom_target(mimalloc_override ALL DEPENDS ${mimalloc_output})

    add_library(mimalloc IMPORTED SHARED STATIC)
    add_dependencies(mimalloc mimalloc_override)
    set_target_properties(mimalloc PROPERTIES IMPORTED_LOCATION "${mimalloc_output_dir}${mimalloc_output}.lib")

    # copy the dlls into the directory where setup.py will look for them
    add_custom_command(TARGET mimalloc_override POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${mimalloc_output_dir}mimalloc-override.dll ${mimalloc_deps}
                       ${mimalloc_wheel_dir}
                   )

  else()
    set(MI_BUILD_TESTS OFF CACHE BOOL "Build mimalloc tests" FORCE)
    add_subdirectory(${mimalloc_root_dir} EXCLUDE_FROM_ALL)
    set_target_properties(mimalloc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    target_compile_definitions(mimalloc PUBLIC MI_USE_CXX=ON)

    # copy the dll into the directory where setup.py will look for it
    get_target_property(mimalloc_output_name mimalloc OUTPUT_NAME)
    install(TARGETS mimalloc DESTINATION ${mimalloc_wheel_dir})
  endif()
  
  # TODO: enable linking once mimalloc has been integrated with an allocator class
  # target_link_libraries(onnxruntime_providers mimalloc)
endif()

if (onnxruntime_USE_CUDA)
  file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
  )
  file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cuh"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})

  # disable contrib ops conditionally
  if(onnxruntime_DISABLE_CONTRIB_OPS)
    add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  else()
    add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs} ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
  endif()

  if (UNIX)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
  endif()
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_cuda ${onnxruntime_EXTERNAL_DEPENDENCIES} ${onnxruntime_tvm_dependencies})
  target_include_directories(onnxruntime_providers_cuda PRIVATE ${ONNXRUNTIME_ROOT} ${PROJECT_SOURCE_DIR}/external/cub ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cuda  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES FOLDER "ONNXRuntime")
  if (WIN32)
    # *.cu cannot use PCH
    foreach(src_file ${onnxruntime_providers_cuda_cc_srcs})
      set_source_files_properties(${src_file}
        PROPERTIES
        COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
    endforeach()
    if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
      foreach(src_file ${onnxruntime_cuda_contrib_ops_cc_srcs})
        set_source_files_properties(${src_file}
          PROPERTIES
          COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
      endforeach()
    endif()
    set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
      PROPERTIES
      COMPILE_FLAGS "/Yccuda_pch.h"
    )
    # disable a warning from the CUDA headers about unreferenced local functions
    #target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505)
    if (onnxruntime_USE_TVM)
      target_compile_options(onnxruntime_providers_cuda PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    endif()
  endif()
endif()

if (onnxruntime_USE_MKLDNN)
  file(GLOB_RECURSE onnxruntime_providers_mkldnn_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/mkldnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/mkldnn/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_mkldnn_cc_srcs})
  add_library(onnxruntime_providers_mkldnn ${onnxruntime_providers_mkldnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_mkldnn onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_mkldnn ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_mkldnn PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${MKLDNN_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/mkldnn  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT=1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  include_directories(${PROJECT_SOURCE_DIR}/external/protobuf)
  set(CUDA_INCLUDE_DIRS ${onnxruntime_CUDA_HOME}/include)
  set(TENSORRT_ROOT ${onnxruntime_TENSORRT_HOME})
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx)
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if (WIN32)
    set(OLD_CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267 /wd4099 /wd4551 /wd4505 /wd4515 /wd4706 /wd4456 /wd2220")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4701 /wd4805")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include algorithm")
    set(PROTOBUF_LIBRARY libprotobuf)
    set(DISABLED_WARNINGS_FOR_TRT /wd4267 /wd4244 /wd4996)
  endif()
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)
  add_subdirectory(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  if (WIN32)
    set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
    unset(PROTOBUF_LIBRARY)
    unset(OLD_CMAKE_CXX_FLAGS)
    unset(OLD_CMAKE_CUDA_FLAGS)
    set_target_properties(nvonnxparser PROPERTIES LINK_FLAGS "/ignore:4199")
    set_target_properties(nvonnxparser_runtime PROPERTIES LINK_FLAGS "/ignore:4199")
    set_target_properties(trt_onnxify PROPERTIES LINK_FLAGS "/ignore:4199")
    target_compile_definitions(trt_onnxify PRIVATE ONNXIFI_BUILD_LIBRARY=1)
    target_sources(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_sources(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_include_directories(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_include_directories(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_compile_options(nvonnxparser_static PRIVATE /FIio.h)
    target_compile_options(nvonnxparser PRIVATE /FIio.h)
    target_compile_options(trt_onnxify PRIVATE /FIio.h)
    target_compile_options(onnx2trt PRIVATE /FIio.h)
    target_compile_options(getSupportedAPITest PRIVATE /FIio.h)
  endif()
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  include_directories(${TENSORRT_INCLUDE_DIR})
  set(trt_link_libs cudnn ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})
  set(onnxparser_link_libs nvonnxparser_static nvonnxparser_plugin)

  file(GLOB_RECURSE onnxruntime_providers_tensorrt_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tensorrt_cc_srcs})
  add_library(onnxruntime_providers_tensorrt ${onnxruntime_providers_tensorrt_cc_srcs})
  target_link_libraries(onnxruntime_providers_tensorrt ${onnxparser_link_libs} ${trt_link_libs})
  onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_tensorrt ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/tensorrt  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_tensorrt PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_tensorrt PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_tensorrt INTERFACE /wd4996)
  endif()
endif()

if (onnxruntime_USE_NGRAPH)
  include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
  file(GLOB_RECURSE onnxruntime_providers_ngraph_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/ngraph/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/ngraph/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_ngraph_cc_srcs})
  add_library(onnxruntime_providers_ngraph ${onnxruntime_providers_ngraph_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_ngraph onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_ngraph ngraph onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_ngraph PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_ngraph PRIVATE ${ONNXRUNTIME_ROOT} ${ngraph_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_ngraph PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT MSVC)
    target_compile_options(onnxruntime_providers_ngraph PRIVATE "SHELL:-Wformat" "SHELL:-Wformat-security" "SHELL:-fstack-protector-strong" "SHELL:-D_FORTIFY_SOURCE=2")
    target_link_options(onnxruntime_providers_ngraph PRIVATE "LINKER:-z, noexecstack " "LINKER:-z relro" "LINKER:-z now" "LINKER:-pie")
  endif()
endif()

if (onnxruntime_USE_OPENVINO)
  file(GLOB_RECURSE onnxruntime_providers_openvino_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cc"
  )
  file(GLOB_RECURSE onnxruntime_providers_openvino_py_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/openvino_mo/*.py"
  )

  # Below variables point to directories within the OpenVINO installation directory
  # whose value is set in INTEL_CVSDK_DIR variable by running the setupvars.sh script
if (onnxruntime_USE_OPENVINO_BINARY)
  if ($ENV{INTEL_CVSDK_DIR} MATCHES "2019.1")
     message($ENV{INTEL_CVSDK_DIR})
     set(OPENVINO_INCLUDE_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/include)
     set(OPENVINO_TBB_INCLUDE_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/include)
    if(WIN32)
     set(OPENVINO_LIB_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/Release)
     set(OPENVINO_TBB_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/Release)
     set(OPENVINO_MKL_TINY_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/bin/intel64/Release)
	 
    else()
     set(OPENVINO_LIB_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/)
     set(OPENVINO_TBB_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/lib)
     set(OPENVINO_MKL_TINY_DIR $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/mkltiny_lnx/lib)
    endif()
  endif()
  if ($ENV{INTEL_CVSDK_DIR} MATCHES "2018.5")
     set(OPENVINO_INCLUDE_DIR $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/include)
     set(OPENVINO_LIB_DIR $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/)
  endif()
endif()

  find_package(PythonLibs REQUIRED)
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})
  add_library(onnxruntime_providers_openvino ${onnxruntime_providers_openvino_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_openvino onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_openvino ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_openvino PROPERTIES FOLDER "ONNXRuntime")
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES LINKER_LANGUAGE CXX)
  if (WIN32)
    target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${OPENVINO_INCLUDE_DIR} ${OPENVINO_TBB_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${PYTHONPATH})
   #${pybind11_INCLUDE_DIRS}
  else()
    target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${OPENVINO_INCLUDE_DIR} ${OPENVINO_TBB_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  endif()
  
  if ($ENV{INTEL_CVSDK_DIR} MATCHES "2019.1")
   if (WIN32)
    
     string(REPLACE "include" "libs" PYTHON_LIB ${PYTHON_INCLUDE_DIRS})	
     set(PYTHON_LIBRARIES ${PYTHON_LIB})
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267 /wd4099 /wd4551 /wd4505 /wd4515 /wd4706 /wd4456 /w")
     link_directories(onnxruntime_providers_openvino -linference_engine ${PYTHON_LIBRARIES} ${OPENVINO_LIB_DIR} ${OPENVINO_TBB_DIR} ${OPENVINO_MKL_TINY_DIR} ${PYTHONPATH})
     target_link_libraries(onnxruntime_providers_openvino $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/intel64/Release/inference_engine.lib)
     file(COPY ${onnxruntime_providers_openvino_py_srcs} DESTINATION ${onnxruntime_BINARY_DIR}/${CMAKE_BUILD_TYPE})
   else()
      link_directories(onnxruntime_providers_openvino ${OPENVINO_LIB_DIR} ${OPENVINO_TBB_DIR} ${OPENVINO_MKL_TINY_DIR})
      target_link_libraries(onnxruntime_providers_openvino -linference_engine -ltbb ${PYTHON_LIBRARIES})
      file(COPY ${onnxruntime_providers_openvino_py_srcs} DESTINATION ${onnxruntime_BINARY_DIR})
   endif()
  endif()
  if ($ENV{INTEL_CVSDK_DIR} MATCHES "2018.5")
    link_directories(onnxruntime_providers_openvino ${OPENVINO_LIB_DIR})
    target_link_libraries(onnxruntime_providers_openvino -linference_engine ${PYTHON_LIBRARIES})
  endif()
  file(COPY ${onnxruntime_providers_openvino_py_srcs} DESTINATION ${onnxruntime_BINARY_DIR})
endif()

if (onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR=1)

  if (NOT onnxruntime_USE_TVM)
    message(FATAL_ERROR "onnxruntime_USE_TVM required for onnxruntime_USE_NUPHAR")
  endif()

  if (NOT onnxruntime_USE_LLVM)
    message(FATAL_ERROR "onnxruntime_USE_LLVM required for onnxruntime_USE_NUPHAR")
  endif()

  include(onnxruntime_nuphar_extern.cmake)

  file(GLOB_RECURSE onnxruntime_providers_nuphar_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.cc"
  )

  # following files required different build flag for AVX2 in separate onnxruntime_nuphar_extern.cmake file
  list (REMOVE_ITEM onnxruntime_providers_nuphar_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.cc")
  list (REMOVE_ITEM onnxruntime_providers_nuphar_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.h")

  if (onnxruntime_USE_MKLML)
    add_definitions(-DNUPHAR_USE_MKL)
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nuphar_cc_srcs})
  add_library(onnxruntime_providers_nuphar ${onnxruntime_providers_nuphar_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nuphar onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_nuphar PRIVATE ${ONNXRUNTIME_ROOT} ${TVM_INCLUDES} ${eigen_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_nuphar PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
  add_dependencies(onnxruntime_providers_nuphar ${onnxruntime_EXTERNAL_DEPENDENCIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/nuphar  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
endif()
 
if (onnxruntime_USE_NNAPI)
  add_definitions(-DUSE_NNAPI=1)
  option(DNN_READ_ONNX "" ON)
  set(DNN_CUSTOM_PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
  option(DNN_CMAKE_INSTALL "" OFF)
  option(DNN_BUILD_BIN "" OFF)
  add_subdirectory(${REPO_ROOT}/cmake/external/DNNLibrary)
  file(GLOB_RECURSE
    onnxruntime_providers_nnapi_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nnapi/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nnapi/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nnapi_cc_srcs})
  add_library(onnxruntime_providers_nnapi ${onnxruntime_providers_nnapi_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nnapi onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite dnnlibrary::dnnlibrary)
  target_link_libraries(onnxruntime_providers_nnapi dnnlibrary::dnnlibrary)
  add_dependencies(onnxruntime_providers_nnapi
    dnnlibrary::dnnlibrary
    onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  # Header files of DNNLibrary requires C++17, fortunately, all modern Android NDKs support C++17
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES CXX_STANDARD 17)
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_nnapi PRIVATE ${ONNXRUNTIME_ROOT} ${nnapi_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_providers_internal.cmake)
endif()

if(onnxruntime_USE_EIGEN_THREADPOOL)
    target_compile_definitions(onnxruntime_providers PUBLIC USE_EIGEN_THREADPOOL)
endif()
