# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(pybind11)

# ---[ Python + Numpy
set(onnxruntime_pybind_srcs_pattern
    "${ONNXRUNTIME_ROOT}/python/*.cc"
    "${ONNXRUNTIME_ROOT}/python/*.h"
)

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND onnxruntime_pybind_srcs_pattern
    "${ORTTRAINING_ROOT}/orttraining/python/*.cc"
    "${ORTTRAINING_ROOT}/orttraining/python/*.h"
  )
endif()

file(GLOB onnxruntime_pybind_srcs CONFIGURE_DEPENDS
  ${onnxruntime_pybind_srcs_pattern}
  )

if(onnxruntime_ENABLE_TRAINING)
  list(REMOVE_ITEM onnxruntime_pybind_srcs  ${ONNXRUNTIME_ROOT}/python/onnxruntime_pybind_module.cc)
endif()

# Add Pytorch as a library.
if (onnxruntime_ENABLE_LAZY_TENSOR)
  # Lazy Tensor requires Pytorch as a library.
  list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_PREBUILT_PYTORCH_PATH})
  # The following line may change ${CUDA_NVCC_FLAGS} and ${CMAKE_CUDA_FLAGS},
  # if Pytorch is built from source.
  # For example, pytorch/cmake/public/cuda.cmake and
  # pytorch/torch/share/cmake/Caffe2/public/cuda.cmake both defines
  # ONNX_NAMESPACE for both CUDA_NVCC_FLAGS and CMAKE_CUDA_FLAGS.
  # Later, this ONNX_NAMESPACE may conflicts with ONNX_NAMESPACE set by ORT.
  find_package(Torch REQUIRED)
  # Let's remove ONNX_NAMESPACE from Torch.
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "-DONNX_NAMESPACE=.+")
  string(REGEX REPLACE "-DONNX_NAMESPACE=.+ " " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

# Support ORT as a backend in Pytorch's LazyTensor.
if (onnxruntime_ENABLE_LAZY_TENSOR)
  file(GLOB onnxruntime_lazy_tensor_extension_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_ROOT}/orttraining/lazy_tensor/*.cc")
  file(GLOB onnxruntime_lazy_tensor_extension_headers CONFIGURE_DEPENDS
    "${ORTTRAINING_ROOT}/orttraining/lazy_tensor/*.h")

  if(NOT MSVC)
    set_source_files_properties(${onnxruntime_lazy_tensor_extension_srcs} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${onnxruntime_lazy_tensor_extension_headers} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()

  list(APPEND onnxruntime_pybind_srcs
              ${onnxruntime_lazy_tensor_extension_srcs})
endif()

# onnxruntime_ENABLE_LAZY_TENSOR and onnxruntime_ENABLE_EAGER_MODE
# need DLPack code to pass tensors cross ORT and Pytorch boundary.
# TODO: consider making DLPack code a standalone library.
if (onnxruntime_ENABLE_LAZY_TENSOR)
  # If DLPack code is not built, add it to ORT's pybind target.
  if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    list(APPEND onnxruntime_pybind_srcs
              "${ORTTRAINING_ROOT}/orttraining/core/framework/torch/dlpack_python.cc")
  endif()
endif()

onnxruntime_add_shared_library_module(onnxruntime_pybind11_state ${onnxruntime_pybind_srcs})

if(MSVC)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  if(onnxruntime_ENABLE_TRAINING)
    target_compile_options(onnxruntime_pybind11_state PRIVATE "/bigobj")
  endif()
endif()
if(HAS_CAST_FUNCTION_TYPE)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "-Wno-cast-function-type")
endif()

# We export symbols using linker and the compiler does not know anything about it
# There is a problem with classes that have pybind types as members.
# See https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
if (NOT MSVC)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "-fvisibility=hidden")
endif()

if(onnxruntime_PYBIND_EXPORT_OPSCHEMA)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE onnxruntime_PYBIND_EXPORT_OPSCHEMA)
endif()

if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings
    target_compile_options(onnxruntime_pybind11_state PRIVATE "/wd4244")
endif()

onnxruntime_add_include_to_target(onnxruntime_pybind11_state Python::Module Python::NumPy)
target_include_directories(onnxruntime_pybind11_state PRIVATE ${ONNXRUNTIME_ROOT} ${pybind11_INCLUDE_DIRS})
if(onnxruntime_USE_CUDA AND onnxruntime_CUDNN_HOME)
    target_include_directories(onnxruntime_pybind11_state PRIVATE ${onnxruntime_CUDNN_HOME}/include)
endif()
if(onnxruntime_USE_CANN)
    target_include_directories(onnxruntime_pybind11_state PRIVATE ${onnxruntime_CANN_HOME}/include)
endif()
if(onnxruntime_USE_ROCM)
  target_compile_options(onnxruntime_pybind11_state PUBLIC -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)
  target_include_directories(onnxruntime_pybind11_state PRIVATE ${onnxruntime_ROCM_HOME}/hipfft/include ${onnxruntime_ROCM_HOME}/include ${onnxruntime_ROCM_HOME}/hiprand/include ${onnxruntime_ROCM_HOME}/rocrand/include ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_pybind11_state PRIVATE ${NCCL_INCLUDE_DIRS})
endif()

if(APPLE)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/python/exported_symbols.lst")
elseif(UNIX)
  if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
    set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/python/version_script_expose_onnx_protobuf.lds -Xlinker --gc-sections")
  else()
    set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/python/version_script.lds -Xlinker --gc-sections")
  endif()
else()
  set(ONNXRUNTIME_SO_LINK_FLAG "-DEF:${ONNXRUNTIME_ROOT}/python/pybind.def")
endif()

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE ENABLE_ATEN)
  target_include_directories(onnxruntime_pybind11_state PRIVATE ${dlpack_SOURCE_DIR}/include)
endif()

if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_pybind11_state PRIVATE ${ORTTRAINING_ROOT})
  target_link_libraries(onnxruntime_pybind11_state PRIVATE onnxruntime_training)
endif()

# Eager mode and LazyTensor are both Pytorch's backends, so their
# dependencies are set together below.
if (onnxruntime_ENABLE_LAZY_TENSOR)
  # Set library dependencies shared by aforementioned backends.

  # todo: this is because the prebuild pytorch may use a different version of protobuf headers.
  # force the build to find the protobuf headers ort using.
  target_include_directories(onnxruntime_pybind11_state PRIVATE
    "${REPO_ROOT}/cmake/external/protobuf/src"
    ${TORCH_INCLUDE_DIRS})

  # For eager mode, torch build has a mkl dependency from torch's cmake config,
  # Linking to torch libraries to avoid this unnecessary mkl dependency.
  target_include_directories(onnxruntime_pybind11_state PRIVATE "${TORCH_INSTALL_PREFIX}/include" "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
  find_library(LIBTORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib")
  find_library(LIBTORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib")
  find_library(LIBC10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib")
  # Explicitly link torch_python to workaround https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  target_link_libraries(onnxruntime_pybind11_state PRIVATE  ${LIBTORCH_LIBRARY} ${LIBTORCH_CPU_LIBRARY} ${LIBC10_LIBRARY} ${TORCH_PYTHON_LIBRARY})
  if (onnxruntime_USE_CUDA)
    find_library(LIBTORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
    find_library(LIBC10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(onnxruntime_pybind11_state PRIVATE ${LIBTORCH_CUDA_LIBRARY} ${LIBC10_CUDA_LIBRARY})
  endif()


  if (MSVC)
    target_compile_options(onnxruntime_pybind11_state PRIVATE "/wd4100" "/wd4324" "/wd4458" "/wd4127" "/wd4193" "/wd4624" "/wd4702")
    target_compile_options(onnxruntime_pybind11_state PRIVATE "/bigobj" "/wd4275" "/wd4244" "/wd4267" "/wd4067")
  endif()
endif()

target_link_libraries(onnxruntime_pybind11_state PRIVATE
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_TVM}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_COREML}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
    ${PROVIDERS_QNN}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
    onnxruntime_flatbuffers
    ${pybind11_lib}
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime_pybind11_state PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

set(onnxruntime_pybind11_state_dependencies
    ${onnxruntime_EXTERNAL_DEPENDENCIES}
    ${pybind11_dep}
)
set_property(TARGET onnxruntime_pybind11_state APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_DELAYLOAD_FLAGS})
add_dependencies(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_dependencies})

if (MSVC)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "${ONNXRUNTIME_SO_LINK_FLAG}")
  # if MSVC, pybind11 undefines _DEBUG in pybind11/detail/common.h, which causes the pragma in pyconfig.h
  # from the python installation to require the release version of the lib
  # e.g. from a python 3.10 install:
  #                       if defined(_DEBUG)
  #                               pragma comment(lib,"python310_d.lib")
  #                       elif defined(Py_LIMITED_API)
  #                               pragma comment(lib,"python3.lib")
  #                       else
  #                               pragma comment(lib,"python310.lib")
  #                       endif /* _DEBUG */
  #
  # See https://github.com/pybind/pybind11/issues/3403 for more background info.
  #
  # Explicitly use the release version of the python library to make the project file consistent with this.
  target_link_libraries(onnxruntime_pybind11_state PRIVATE ${Python_LIBRARY_RELEASE})
elseif (APPLE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "${ONNXRUNTIME_SO_LINK_FLAG} -undefined dynamic_lookup")
  set_target_properties(onnxruntime_pybind11_state PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  set_property(TARGET onnxruntime_pybind11_state APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\\$ORIGIN")
endif()

if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  set(onnxruntime_CUSTOM_EXTERNAL_LIBRARIES "${onnxruntime_EXTERNAL_LIBRARIES}")
  list(FIND  onnxruntime_CUSTOM_EXTERNAL_LIBRARIES onnx ONNX_INDEX)
  list(FIND onnxruntime_CUSTOM_EXTERNAL_LIBRARIES ${PROTOBUF_LIB} PROTOBUF_INDEX)
  MATH(EXPR PROTOBUF_INDEX_NEXT "${PROTOBUF_INDEX} + 1")
  if (ONNX_INDEX GREATER_EQUAL 0 AND PROTOBUF_INDEX GREATER_EQUAL 0)
    # Expect protobuf to follow onnx due to dependence
    list(INSERT  onnxruntime_CUSTOM_EXTERNAL_LIBRARIES ${ONNX_INDEX} "-Wl,--no-as-needed")
    list(INSERT onnxruntime_CUSTOM_EXTERNAL_LIBRARIES ${PROTOBUF_INDEX_NEXT} "-Wl,--as-needed")
  else()
    message(FATAL_ERROR "Required external libraries onnx and protobuf are not found in onnxruntime_EXTERNAL_LIBRARIES")
  endif()
  target_link_libraries(onnxruntime_pybind11_state PRIVATE ${onnxruntime_CUSTOM_EXTERNAL_LIBRARIES})
else()
  target_link_libraries(onnxruntime_pybind11_state PRIVATE ${onnxruntime_EXTERNAL_LIBRARIES})
endif()

set_target_properties(onnxruntime_pybind11_state PROPERTIES PREFIX "")
set_target_properties(onnxruntime_pybind11_state PROPERTIES FOLDER "ONNXRuntime")
if(onnxruntime_ENABLE_LTO)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
endif()
if (MSVC)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".so")
endif()

# Generate version_info.py in Windows build.
# Has to be done before onnxruntime_python_srcs is set.
if (WIN32)
  set(VERSION_INFO_FILE "${ONNXRUNTIME_ROOT}/python/version_info.py")

  if (onnxruntime_USE_CUDA)
    file(WRITE "${VERSION_INFO_FILE}" "use_cuda = True\n")
    if(onnxruntime_CUDNN_HOME)
      file(GLOB CUDNN_DLL_PATH "${onnxruntime_CUDNN_HOME}/bin/cudnn64_*.dll")
      if (NOT CUDNN_DLL_PATH)
        message(FATAL_ERROR "cuDNN not found in ${onnxruntime_CUDNN_HOME}")
      endif()
    else()
      file(GLOB CUDNN_DLL_PATH "${onnxruntime_CUDA_HOME}/bin/cudnn64_*.dll")
      if (NOT CUDNN_DLL_PATH)
        message(FATAL_ERROR "cuDNN not found in ${onnxruntime_CUDA_HOME}")
      endif()
    endif()
    get_filename_component(CUDNN_DLL_NAME ${CUDNN_DLL_PATH} NAME_WE)
    string(REPLACE "cudnn64_" "" CUDNN_VERSION "${CUDNN_DLL_NAME}")
    if(NOT onnxruntime_CUDA_VERSION)
      message("Reading json file ${onnxruntime_CUDA_HOME}/version.json")
      set(CUDA_SDK_JSON_FILE_PATH "${onnxruntime_CUDA_HOME}/version.json")
      file(READ ${CUDA_SDK_JSON_FILE_PATH} CUDA_SDK_JSON_CONTENT)
      string(JSON onnxruntime_CUDA_VERSION GET ${CUDA_SDK_JSON_CONTENT} "cuda" "version")
      message("onnxruntime_CUDA_VERSION=${onnxruntime_CUDA_VERSION}")
    endif()
    file(APPEND "${VERSION_INFO_FILE}"
      "cuda_version = \"${onnxruntime_CUDA_VERSION}\"\n"
      "cudnn_version = \"${CUDNN_VERSION}\"\n"
    )
  else()
    file(WRITE "${VERSION_INFO_FILE}" "use_cuda = False\n")
  endif()

  if ("${MSVC_TOOLSET_VERSION}" STREQUAL "142")
    file(APPEND "${VERSION_INFO_FILE}" "vs2019 = True\n")
  else()
    file(APPEND "${VERSION_INFO_FILE}" "vs2019 = False\n")
  endif()
endif()

file(GLOB onnxruntime_backend_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/backend/*.py"
)

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB onnxruntime_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/*.py"
    "${ORTTRAINING_SOURCE_DIR}/python/*.py"
  )
else()
  file(GLOB onnxruntime_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/*.py"
  )
endif()

# Generate _pybind_state.py from _pybind_state.py.in replacing macros with either setdlopenflags or ""
if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  set(ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL "sys.setdlopenflags(os.RTLD_GLOBAL|os.RTLD_NOW|os.RTLD_DEEPBIND)")
  set(ONNXRUNTIME_SETDLOPENFLAGS_LOCAL  "sys.setdlopenflags(os.RTLD_LOCAL|os.RTLD_NOW|os.RTLD_DEEPBIND)")
else()
  set(ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL "")
  set(ONNXRUNTIME_SETDLOPENFLAGS_LOCAL  "")
endif()

if (onnxruntime_ENABLE_LAZY_TENSOR)
  # Import torch so that onnxruntime's pybind can see its DLLs.
  set(ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS "import torch")
else()
  set(ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS "")
endif()

configure_file(${ONNXRUNTIME_ROOT}/python/_pybind_state.py.in
               ${CMAKE_BINARY_DIR}/onnxruntime/capi/_pybind_state.py)

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB onnxruntime_python_capi_training_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/deprecated/*.py"
  )
  file(GLOB onnxruntime_python_root_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/*.py"
  )
  file(GLOB onnxruntime_python_amp_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/amp/*.py"
  )
  file(GLOB onnxruntime_python_experimental_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/experimental/*.py"
  )
  file(GLOB onnxruntime_python_gradient_graph_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/experimental/gradient_graph/*.py"
  )
  file(GLOB onnxruntime_python_optim_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/optim/*.py"
  )
  file(GLOB onnxruntime_python_torchdynamo_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/torchdynamo/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_experimental_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/experimental/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_experimental_json_config_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/experimental/json_config/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_experimental_hierarchical_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/experimental/hierarchical_ortmodule/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_torch_cpp_ext_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/torch_cpp_extensions/*.py"
  )
  file(GLOB onnxruntime_python_ortmodule_torch_cpp_ext_aten_op_executor_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/torch_cpp_extensions/aten_op_executor/*"
  )
  file(GLOB onnxruntime_python_ortmodule_torch_cpp_ext_torch_interop_utils_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/*"
  )
  file(GLOB onnxruntime_python_ortmodule_torch_cpp_ext_torch_gpu_allocator_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/torch_cpp_extensions/cuda/torch_gpu_allocator/*"
  )
  file(GLOB onnxruntime_python_ortmodule_torch_cpp_ext_fused_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/ortmodule/torch_cpp_extensions/cuda/fused_ops/*"
  )
  file(GLOB onnxruntime_python_utils_data_srcs CONFIGURE_DEPENDS
  "${ORTTRAINING_SOURCE_DIR}/python/training/utils/data/*"
  )
  file(GLOB onnxruntime_python_utils_hooks_srcs CONFIGURE_DEPENDS
  "${ORTTRAINING_SOURCE_DIR}/python/training/utils/hooks/*"
  )
  if (onnxruntime_ENABLE_TRAINING_APIS)
    file(GLOB onnxruntime_python_onnxblock_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/onnxblock/*"
    )
    file(GLOB onnxruntime_python_onnxblock_loss_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/onnxblock/loss/*"
    )
    file(GLOB onnxruntime_python_api_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/api/*"
    )
    file(GLOB onnxruntime_python_onnxblock_optim_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/onnxblock/optim/*"
    )
  endif()
else()
  file(GLOB onnxruntime_python_capi_training_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/training/*.py"
  )
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  file(GLOB onnxruntime_python_test_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/*.py"
      "${ORTTRAINING_SOURCE_DIR}/test/python/*.py"
      "${ORTTRAINING_SOURCE_DIR}/test/python/*.json"
  )
  file(GLOB onnxruntime_python_quantization_test_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/quantization/*.py"
  )
  file(GLOB onnxruntime_python_transformers_test_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/transformers/*.py"
  )
  file(GLOB onnxruntime_python_transformers_testdata_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/transformers/test_data/models/*.onnx"
  )
  file(GLOB onnxruntime_python_transformers_testdata_whisper CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/transformers/test_data/models/whisper/*.onnx"
  )
endif()

file(GLOB onnxruntime_python_tools_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/*.py"
)
file(GLOB onnxruntime_python_quantization_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/quantization/*.py"
)
file(GLOB onnxruntime_python_quantization_operators_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/quantization/operators/*.py"
)
file(GLOB onnxruntime_python_quantization_cal_table_flatbuffers_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/quantization/CalTableFlatBuffers/*.py"
)
file(GLOB onnxruntime_python_transformers_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/*.py"
)
file(GLOB onnxruntime_python_transformers_models_bart_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/bart/*.py"
)
file(GLOB onnxruntime_python_transformers_models_bert_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/bert/*.py"
)
file(GLOB onnxruntime_python_transformers_models_gpt2_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/gpt2/*.py"
)
file(GLOB onnxruntime_python_transformers_models_longformer_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/longformer/*.py"
)
file(GLOB onnxruntime_python_transformers_models_stable_diffusion_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/stable_diffusion/*.py"
)
file(GLOB onnxruntime_python_transformers_models_t5_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/t5/*.py"
)
file(GLOB onnxruntime_python_transformers_models_whisper_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/models/whisper/*.py"
)
file(GLOB onnxruntime_python_datasets_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.py"
)
file(GLOB onnxruntime_python_datasets_data CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.pb"
    "${ONNXRUNTIME_ROOT}/python/datasets/*.onnx"
)

# ORT Mobile helpers to convert ONNX model to ORT format, analyze model for suitability in mobile scenarios,
# and assist with export from PyTorch.
set(onnxruntime_mobile_util_srcs
    ${REPO_ROOT}/tools/python/util/check_onnx_model_mobile_usability.py
    ${REPO_ROOT}/tools/python/util/convert_onnx_models_to_ort.py
    ${REPO_ROOT}/tools/python/util/file_utils.py
    ${REPO_ROOT}/tools/python/util/logger.py
    ${REPO_ROOT}/tools/python/util/make_dynamic_shape_fixed.py
    ${REPO_ROOT}/tools/python/util/onnx_model_utils.py
    ${REPO_ROOT}/tools/python/util/optimize_onnx_model.py
    ${REPO_ROOT}/tools/python/util/pytorch_export_helpers.py
    ${REPO_ROOT}/tools/python/util/reduced_build_config_parser.py
    ${REPO_ROOT}/tools/python/util/update_onnx_opset.py
)
file(GLOB onnxruntime_ort_format_model_srcs CONFIGURE_DEPENDS
    ${REPO_ROOT}/tools/python/util/ort_format_model/*.py
)
file(GLOB onnxruntime_mobile_helpers_srcs CONFIGURE_DEPENDS
    ${REPO_ROOT}/tools/python/util/mobile_helpers/*.py
    ${REPO_ROOT}/tools/ci_build/github/android/mobile_package.required_operators.config
    ${REPO_ROOT}/tools/ci_build/github/android/nnapi_supported_ops.md
    ${REPO_ROOT}/tools/ci_build/github/apple/coreml_supported_ops.md
)
file(GLOB onnxruntime_qdq_helper_srcs CONFIGURE_DEPENDS
    ${REPO_ROOT}/tools/python/util/qdq_helpers/*.py
)

if (onnxruntime_USE_OPENVINO)
  file(GLOB onnxruntime_python_openvino_python_srcs CONFIGURE_DEPENDS
    ${REPO_ROOT}/tools/python/util/add_openvino_win_libs.py
  )
endif()

set(build_output_target onnxruntime_common)
if(NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
add_custom_command(
  TARGET onnxruntime_pybind11_state POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/backend
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/mobile_helpers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/qdq_helpers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/ort_format_model
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/ort_format_model/ort_flatbuffers_py
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/bart
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/bert
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/gpt2
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/longformer
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/stable_diffusion
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/t5
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/whisper
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/operators
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/CalTableFlatBuffers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/quantization
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/transformers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/transformers/test_data/models
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/transformers/test_data/models/whisper
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/eager_test
  COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_ROOT}/__init__.py
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/ThirdPartyNotices.txt
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/docs/Privacy.md
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/LICENSE
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_backend_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/backend/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/onnxruntime/capi/_pybind_state.py
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_capi_training_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training/
  COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:onnxruntime_pybind11_state>
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_data}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_tools_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_mobile_util_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/
  # append the /tools/python/utils imports to the __init__.py that came from /onnxruntime/tools.
  # we're aggregating scripts from two different locations, and only include selected functionality from
  # /tools/python/util. due to that we take the full __init__.py from /onnxruntime/tools and append
  # the required content from /tools/python/util/__init__append.py.
  COMMAND ${CMAKE_COMMAND} -E cat
      ${REPO_ROOT}/tools/python/util/__init__append.py >>
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/__init__.py
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_qdq_helper_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/qdq_helpers/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_mobile_helpers_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/mobile_helpers/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_ort_format_model_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/ort_format_model/
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${ONNXRUNTIME_ROOT}/core/flatbuffers/ort_flatbuffers_py
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/ort_format_model/ort_flatbuffers_py
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_quantization_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_quantization_operators_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/operators/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_quantization_cal_table_flatbuffers_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/CalTableFlatBuffers/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_bart_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/bart/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_bert_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/bert/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_gpt2_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/gpt2/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_longformer_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/longformer/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_stable_diffusion_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/stable_diffusion/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_t5_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/t5/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_models_whisper_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/models/whisper/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/VERSION_NUMBER
      $<TARGET_FILE_DIR:${build_output_target}>
)

if (onnxruntime_USE_OPENVINO)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_openvino_python_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/
  )
endif()

if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/external/include/
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_DIR:${build_output_target}>/include/google
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/external/include/google
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_DIR:${build_output_target}>/external/onnx/onnx
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/external/include/onnx
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${ORTTRAINING_ROOT}/orttraining/test/external_custom_ops
        $<TARGET_FILE_DIR:${build_output_target}>/external_custom_ops
    )
endif()

if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android"
                                  AND NOT onnxruntime_USE_ROCM
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:onnxruntime_providers_shared>
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_quantization_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/quantization/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_transformers_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/transformers/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_transformers_testdata_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/transformers/test_data/models/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_transformers_testdata_whisper}
        $<TARGET_FILE_DIR:${build_output_target}>/transformers/test_data/models/whisper/
  )
endif()

if (onnxruntime_BUILD_UNIT_TESTS AND onnxruntime_ENABLE_EAGER_MODE)
  file(GLOB onnxruntime_eager_test_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_ROOT}/orttraining/eager/test/*.py"
  )
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_eager_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/eager_test/
  )
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/amp
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/experimental
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/experimental/gradient_graph
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/optim
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/torchdynamo
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental/json_config
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental/hierarchical_ortmodule
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/aten_op_executor
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cuda/torch_gpu_allocator
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cuda/fused_ops
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/utils/data/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/utils/hooks/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_capi_training_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_root_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_amp_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/amp/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_experimental_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/experimental/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_gradient_graph_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/experimental/gradient_graph/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_optim_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/optim/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_torchdynamo_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/torchdynamo/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_experimental_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_experimental_json_config_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental/json_config/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_experimental_hierarchical_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/experimental/hierarchical_ortmodule/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_torch_cpp_ext_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_torch_cpp_ext_aten_op_executor_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/aten_op_executor/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_torch_cpp_ext_torch_interop_utils_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cpu/torch_interop_utils/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_torch_cpp_ext_torch_gpu_allocator_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cuda/torch_gpu_allocator/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_ortmodule_torch_cpp_ext_fused_ops_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/ortmodule/torch_cpp_extensions/cuda/fused_ops/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_utils_data_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/utils/data/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_utils_hooks_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/utils/hooks/
  )
  if (onnxruntime_ENABLE_TRAINING_APIS)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock/loss
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock/optim
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/api
      COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_onnxblock_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock/
      COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_onnxblock_loss_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock/loss/
      COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_onnxblock_optim_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/onnxblock/optim/
      COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_api_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/api/

    )
  endif()
endif()

if (onnxruntime_USE_DNNL)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${DNNL_DLL_PATH} $<TARGET_FILE:onnxruntime_providers_dnnl>
        $<TARGET_FILE:onnxruntime_providers_shared>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_TENSORRT)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_tensorrt>
        $<TARGET_FILE:onnxruntime_providers_shared>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_MIGRAPHX)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_migraphx>
        $<TARGET_FILE:onnxruntime_providers_shared>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_OPENVINO)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:onnxruntime_providers_openvino>
          $<TARGET_FILE:onnxruntime_providers_shared>
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (DEFINED ENV{OPENVINO_MANYLINUX})
    file(GLOB onnxruntime_python_openvino_python_srcs CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/core/providers/openvino/scripts/*"
    )

    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          ${onnxruntime_python_openvino_python_srcs}
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (onnxruntime_USE_CUDA)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:onnxruntime_providers_cuda>
          $<TARGET_FILE:onnxruntime_providers_shared>
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (onnxruntime_USE_CANN)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:onnxruntime_providers_cann>
          $<TARGET_FILE:onnxruntime_providers_shared>
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (onnxruntime_USE_ROCM)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:onnxruntime_providers_rocm>
          $<TARGET_FILE:onnxruntime_providers_shared>
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (onnxruntime_USE_TVM)
  file(GLOB onnxruntime_python_providers_tvm_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/providers/tvm/*.py"
  )
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/providers
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/providers/tvm
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_providers_tvm_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/providers/tvm
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_tvm>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )

  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
      WORKING_DIRECTORY ${tvm_SOURCE_DIR}/python
      COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel
    )

  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${Python_EXECUTABLE}
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/providers/tvm/extend_python_file.py
          --target_file $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/_ld_preload.py
  )

endif()

if (onnxruntime_USE_DML)
  if (NOT onnxruntime_USE_CUSTOM_DIRECTML)
    set(dml_shared_lib_path ${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}-win/${DML_SHARED_LIB})
  else()
    set(dml_shared_lib_path ${DML_PACKAGE_DIR}/bin/${DML_SHARED_LIB})
  endif()
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${dml_shared_lib_path}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_nnapi>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_COREML)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_coreml>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

endif()
if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  include(onnxruntime_language_interop_ops.cmake)
endif()
