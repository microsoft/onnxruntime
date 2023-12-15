# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
  find_package(XCTest REQUIRED)
endif()

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND TEST_INC_DIR ${ORTTRAINING_ROOT})
endif()
if (onnxruntime_USE_TVM)
  list(APPEND TEST_INC_DIR ${TVM_INCLUDES})
endif()

set(disabled_warnings)
function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS;TEST_ARGS" ${ARGN})
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    onnxruntime_add_executable(${_UT_TARGET} ${TEST_SRC_DIR}/xctest/orttestmain.m)
  else()
    onnxruntime_add_executable(${_UT_TARGET} ${_UT_SOURCES})
  endif()
  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()

  source_group(TREE ${REPO_ROOT} FILES ${_UT_SOURCES})

  if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings, they are dangerous
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")
  endif()
  if (MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6330>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6330>")
    #Abseil has a lot of C4127/C4324 warnings.
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4127>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4127>")
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4324>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4324>")
  endif()

  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (MSVC)
    # set VS debugger working directory to the test program's directory
    set_target_properties(${_UT_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>)
  endif()

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
            Threads::Threads)
    target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_ONNXRUNTIME_DLL)
  else()
    if(onnxruntime_USE_CUDA)
      #XXX: we should not need to do this. onnxruntime_test_all.exe should not have direct dependency on CUDA DLLs,
      # otherwise it will impact when CUDA DLLs can be unloaded.
      target_link_libraries(${_UT_TARGET} PRIVATE cudart)
    endif()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()

  onnxruntime_add_include_to_target(${_UT_TARGET} date::date flatbuffers::flatbuffers)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  if (onnxruntime_USE_CUDA)
    target_include_directories(${_UT_TARGET} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${onnxruntime_CUDNN_HOME}/include)
    if (onnxruntime_USE_NCCL)
      target_include_directories(${_UT_TARGET} PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
  endif()
  if (onnxruntime_USE_TENSORRT)
    # used for instantiating placeholder TRT builder to mitigate TRT library load/unload overhead
    target_include_directories(${_UT_TARGET} PRIVATE ${TENSORRT_INCLUDE_DIR})
  endif()

  if(MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()

  if (WIN32)
    # include dbghelp in case tests throw an ORT exception, as that exception includes a stacktrace, which requires dbghelp.
    target_link_libraries(${_UT_TARGET} PRIVATE debug dbghelp)

    if (MSVC)
      # warning C6326: Potential comparison of a constant with another constant.
      # Lot of such things came from gtest
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      # Raw new and delete. A lot of such things came from googletest.
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      # "Global initializer calls a non-constexpr function."
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    target_compile_options(${_UT_TARGET} PRIVATE "-Wno-error=uninitialized")
  endif()

  set(TEST_ARGS ${_UT_TEST_ARGS})
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # WebAssembly use a memory file system, so we do not use full path
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<TARGET_FILE_NAME:${_UT_TARGET}>.$<CONFIG>.results.xml")
    else()
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
    endif()
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    # target_sources(${_UT_TARGET} PRIVATE ${TEST_SRC_DIR}/xctest/orttestmain.m)
    set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}
      MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

    xctest_add_bundle(${_UT_TARGET}_xc ${_UT_TARGET}
      ${TEST_SRC_DIR}/xctest/ortxctest.m
      ${TEST_SRC_DIR}/xctest/xcgtest.mm
      ${_UT_SOURCES})
    onnxruntime_configure_target(${_UT_TARGET}_xc)
    if(_UT_DYN)
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
              Threads::Threads)
      target_compile_definitions(${_UT_TARGET}_xc PRIVATE USE_ONNXRUNTIME_DLL)
    else()
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
    endif()
    onnxruntime_add_include_to_target(${_UT_TARGET}_xc date::date flatbuffers::flatbuffers)
    target_include_directories(${_UT_TARGET}_xc PRIVATE ${TEST_INC_DIR})
    get_target_property(${_UT_TARGET}_DEFS ${_UT_TARGET} COMPILE_DEFINITIONS)
    target_compile_definitions(${_UT_TARGET}_xc PRIVATE ${_UT_TARGET}_DEFS)

    set_target_properties(${_UT_TARGET}_xc PROPERTIES FOLDER "ONNXRuntimeXCTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}_xc
      MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

    xctest_add_test(xctest.${_UT_TARGET} ${_UT_TARGET}_xc)
  else()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # We might have already executed the following "find_program" code when we build ORT nodejs binding.
      # Then the program is found the result is stored in the variable and the search will not be repeated.
      find_program(NPM_CLI
         NAMES "npm.cmd" "npm"
         DOC "NPM command line client"
         REQUIRED
      )

      if (onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER)
        add_custom_command(TARGET ${_UT_TARGET} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package-lock.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/karma.conf.js $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${NPM_CLI} ci
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )

        set(TEST_NPM_FLAGS)
        if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
          list(APPEND TEST_NPM_FLAGS "--wasm-threads")
        endif()
        add_test(NAME ${_UT_TARGET}
          COMMAND ${NPM_CLI} test -- ${TEST_NPM_FLAGS} --entry=${_UT_TARGET} ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      else()
        set(TEST_NODE_FLAGS)
        if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
          list(APPEND TEST_NODE_FLAGS "--experimental-wasm-threads")
        endif()
        if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
          list(APPEND TEST_NODE_FLAGS "--experimental-wasm-simd")
        endif()

        # prefer Node from emsdk so the version is more deterministic
        if (DEFINED ENV{EMSDK_NODE})
          set(NODE_EXECUTABLE $ENV{EMSDK_NODE})
        else()
          # warning as we don't know what node version is being used and whether things like the TEST_NODE_FLAGS
          # will be valid. e.g. "--experimental-wasm-simd" is not valid with node v20 or later.
          message(WARNING "EMSDK_NODE environment variable was not set. Falling back to system `node`.")
          set(NODE_EXECUTABLE node)
        endif()

        add_test(NAME ${_UT_TARGET}
          COMMAND ${NODE_EXECUTABLE} ${TEST_NODE_FLAGS} ${_UT_TARGET}.js ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      endif()
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TARGET} PROPERTIES TIMEOUT 7200)
    else()
      add_test(NAME ${_UT_TARGET}
        COMMAND ${_UT_TARGET} ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
      )
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TARGET} PROPERTIES TIMEOUT 7200)
    endif()
  endif()
endfunction(AddTest)

# general program entrypoint for C++ unit tests
set(onnxruntime_unittest_main_src "${TEST_SRC_DIR}/unittest_main/test_main.cc")

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
)

file(GLOB onnxruntime_test_quantiztion_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/quantization/*.cc"
  "${TEST_SRC_DIR}/quantization/*.h"
)

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/ir/*.cc"
    "${TEST_SRC_DIR}/ir/*.h"
    )

  file(GLOB onnxruntime_test_optimizer_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/optimizer/*.cc"
    "${TEST_SRC_DIR}/optimizer/*.h"
    )

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/framework/*.cc"
    "${TEST_SRC_DIR}/framework/*.h"
    "${TEST_SRC_DIR}/platform/*.cc"
    )

else()  # minimal and/or reduced ops build

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/*.cc"
    )

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
    list(APPEND onnxruntime_test_framework_src_patterns
      "${TEST_SRC_DIR}/framework/ort_model_only_test.cc"
    )
  endif()

  if (NOT onnxruntime_MINIMAL_BUILD)
    file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/ir/*.cc"
      "${TEST_SRC_DIR}/ir/*.h"
      )
  endif()
endif()

if((NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
   AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_optimizer_src
       "${TEST_SRC_DIR}/optimizer/runtime_optimization/graph_runtime_optimization_test.cc")
endif()

file(GLOB onnxruntime_test_training_src
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/framework/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.cc"
  )

# TODO (baijumeswani): Remove the minimal build check here.
#                      The training api tests should be runnable even on a minimal build.
#                      This requires converting all the *.onnx files to ort format.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_APIS)
    file(GLOB onnxruntime_test_training_api_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.h"
      )
  endif()
endif()

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  if(onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
  endif()

  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/providers/*.h"
    "${TEST_SRC_DIR}/providers/*.cc"
    "${TEST_SRC_DIR}/opaque_api/test_opaque_api.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.h"
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
  )

  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND onnxruntime_test_providers_src_patterns
      "${TEST_SRC_DIR}/contrib_ops/*.h"
      "${TEST_SRC_DIR}/contrib_ops/*.cc"
      "${TEST_SRC_DIR}/contrib_ops/math/*.h"
      "${TEST_SRC_DIR}/contrib_ops/math/*.cc")
  endif()

else()
  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
    # TODO: Add anything that is needed for testing a minimal build
  )
endif()

file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS ${onnxruntime_test_providers_src_patterns})

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_cpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cpu/*"
    )
endif()

if(onnxruntime_DISABLE_ML_OPS)
  list(FILTER onnxruntime_test_providers_cpu_src EXCLUDE REGEX ".*/ml/.*")
endif()

list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

if (onnxruntime_USE_CUDA AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB onnxruntime_test_providers_cuda_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_src})

  if (onnxruntime_USE_CUDA_NHWC_OPS)
    file(GLOB onnxruntime_test_providers_cuda_nhwc_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/providers/cuda/nhwc/*.cc"
    )
    list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_nhwc_src})
  endif()
endif()

if (onnxruntime_USE_CANN)
  file(GLOB_RECURSE onnxruntime_test_providers_cann_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cann/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cann_src})
endif()

# Disable training ops test for minimal build as a lot of these depend on loading an onnx model.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE orttraining_test_trainingops_cpu_src CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/compare_provider_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/function_op_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/*"
      )

    if (NOT onnxruntime_ENABLE_TRAINING)
      list(REMOVE_ITEM orttraining_test_trainingops_cpu_src
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/tensorboard/summary_op_test.cc"
        )
    endif()

    list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cpu_src})

    if (onnxruntime_USE_CUDA OR onnxruntime_USE_ROCM)
      file(GLOB_RECURSE orttraining_test_trainingops_cuda_src CONFIGURE_DEPENDS
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cuda/*"
        )
      list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cuda_src})
    endif()
  endif()
endif()

if (onnxruntime_USE_DNNL)
  file(GLOB_RECURSE onnxruntime_test_providers_dnnl_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/dnnl/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_dnnl_src})
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  file(GLOB_RECURSE onnxruntime_test_providers_nnapi_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/nnapi/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_nnapi_src})
endif()

if (onnxruntime_USE_RKNPU)
  file(GLOB_RECURSE onnxruntime_test_providers_rknpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/rknpu/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_rknpu_src})
endif()

if (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_internal_testing_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/internal_testing/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_internal_testing_src})
endif()

set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${TEST_SRC_DIR}/shared_lib")
set (ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR "${TEST_SRC_DIR}/global_thread_pools")
set (ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR "${TEST_SRC_DIR}/custom_op_registration")
set (ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR "${TEST_SRC_DIR}/logging_apis")

set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_loading.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_ort_format_models.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.cc)

if (NOT onnxruntime_MINIMAL_BUILD)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc)
endif()

if(onnxruntime_RUN_ONNX_TESTS)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
endif()

set (onnxruntime_global_thread_pools_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_main.cc
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_inference.cc)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_test_utils
  onnxruntime_common
)

set(onnxruntime_test_ir_libs
  onnxruntime_test_utils
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_optimizer_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  ${ONNXRUNTIME_MLAS_LIBS}
  onnxruntime_common
  )

set(onnxruntime_test_server_libs
  onnxruntime_test_utils
  onnxruntime_test_utils_for_server
)

if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
endif()

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_CANN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cann)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_DML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dml)
endif()

if(onnxruntime_USE_DNNL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dnnl)
endif()

if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_migraphx)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx onnxruntime_providers_shared)
endif()

if(onnxruntime_USE_ROCM)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rocm)
endif()

if(onnxruntime_USE_COREML)
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml onnxruntime_coreml_proto)
  else()
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml)
  endif()
endif()

if(onnxruntime_USE_ACL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_acl)
endif()

if(onnxruntime_USE_ARMNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_armnn)
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  set(ONNXRUNTIME_INTEROP_TEST_LIBS PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${ONNXRUNTIME_INTEROP_TEST_LIBS}
    ${onnxruntime_libs}
    # CUDA, ROCM, TENSORRT, MIGRAPHX, DNNL, and OpenVINO are dynamically loaded at runtime
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_JS}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_QNN}
    ${PROVIDERS_SNPE}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_COREML}
    # ${PROVIDERS_TVM}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
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
)

if (onnxruntime_ENABLE_TRAINING)
  set(ONNXRUNTIME_TEST_LIBS onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_TEST_LIBS})
endif()

set(onnxruntime_test_providers_libs
    onnxruntime_test_utils
    ${ONNXRUNTIME_TEST_LIBS}
  )

if(onnxruntime_USE_TENSORRT)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/tensorrt/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/tensorrt_execution_provider_utils.h")
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_tensorrt)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_tensorrt onnxruntime_providers_shared)
  list(APPEND onnxruntime_test_providers_libs ${TENSORRT_LIBRARY_INFER})
endif()

if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/migraphx/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/migraphx/migraphx_execution_provider_utils.h")
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_migraphx)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx onnxruntime_providers_shared)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nnapi/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/js/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_js)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_js)
endif()

if(onnxruntime_USE_QNN)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_qnn)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_qnn)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_qnn)
endif()

if(onnxruntime_USE_SNPE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/snpe/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_snpe)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_snpe)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_snpe)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/rknpu/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/coreml/*)
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_coreml onnxruntime_coreml_proto)
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml onnxruntime_coreml_proto)
    list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_coreml onnxruntime_coreml_proto)
  else()
    list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_coreml)
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml)
    list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_coreml)
  endif()
endif()

if(onnxruntime_USE_XNNPACK)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/xnnpack/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_xnnpack)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_xnnpack)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_xnnpack)
endif()

if(onnxruntime_USE_AZURE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/azure/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_azure)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_azure)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_azure)
endif()

if(WIN32)
  if (onnxruntime_USE_TVM)
    list(APPEND disabled_warnings ${DISABLED_WARNINGS_FOR_TVM})
  endif()
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

#This is a small wrapper library that shouldn't use any onnxruntime internal symbols(except onnxruntime_common).
#Because it could dynamically link to onnxruntime. Otherwise you will have two copies of onnxruntime in the same
#process and you won't know which one you are testing.
onnxruntime_add_static_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
if(MSVC)
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
else()
  target_compile_definitions(onnxruntime_test_utils PUBLIC -DNSYNC_ATOMIC_CPP11)
  target_include_directories(onnxruntime_test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  onnxruntime_add_include_to_target(onnxruntime_test_utils nsync::nsync_cpp)
endif()
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_test_utils PRIVATE ${NCCL_INCLUDE_DIRS})
endif()
if (onnxruntime_USE_ROCM)
  target_include_directories(onnxruntime_test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_common onnxruntime_framework onnxruntime_session GTest::gtest GTest::gmock onnx onnx_proto flatbuffers::flatbuffers nlohmann_json::nlohmann_json Boost::mp11 safeint_interface)



if (onnxruntime_USE_DML)
  target_add_dml(onnxruntime_test_utils)
endif()
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE
        ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_test_utils_src})

set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
file(GLOB onnx_test_runner_common_srcs CONFIGURE_DEPENDS
    ${onnx_test_runner_src_dir}/*.h
    ${onnx_test_runner_src_dir}/*.cc)

list(REMOVE_ITEM onnx_test_runner_common_srcs ${onnx_test_runner_src_dir}/main.cc)

onnxruntime_add_static_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
if(MSVC)
  target_compile_options(onnx_test_runner_common PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
else()
  target_compile_definitions(onnx_test_runner_common PUBLIC -DNSYNC_ATOMIC_CPP11)
  target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  onnxruntime_add_include_to_target(onnx_test_runner_common nsync::nsync_cpp)
endif()
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  #TODO: fix the warnings, they are dangerous
  target_compile_options(onnx_test_runner_common PRIVATE "/wd4244")
endif()
onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework
        onnxruntime_test_utils onnx onnx_proto re2::re2 flatbuffers::flatbuffers Boost::mp11 safeint_interface)

add_dependencies(onnx_test_runner_common onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})

set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")

set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_optimizer_src}
        ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src} ${onnxruntime_test_quantiztion_src})

if (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  file(GLOB onnxruntime_test_providers_cuda_ut_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/test_cases/*"
  )
  # onnxruntime_providers_cuda_ut is only for unittests.
  onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_ut ${onnxruntime_test_providers_cuda_ut_src} $<TARGET_OBJECTS:onnxruntime_providers_cuda_obj>)
  config_cuda_provider_shared_module(onnxruntime_providers_cuda_ut)
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda_ut GTest::gtest GTest::gmock)
  target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda_ut)
endif()

set(all_dependencies ${onnxruntime_test_providers_dependencies} )

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND all_tests ${onnxruntime_test_training_src})
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
    list(APPEND all_tests ${onnxruntime_test_training_api_src})
endif()

if (onnxruntime_USE_TVM)
    list(APPEND all_tests ${onnxruntime_test_tvm_src})
endif()

if (onnxruntime_USE_OPENVINO)
  list(APPEND all_tests ${onnxruntime_test_openvino_src})
endif()

# this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
# for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
# list given it's built on top of that library and needs all the same dependencies.
if(WIN32)
  list(APPEND onnxruntime_test_providers_libs Advapi32)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/framework/execution_frame_test.cc"
      "${TEST_SRC_DIR}/framework/inference_session_test.cc"
      "${TEST_SRC_DIR}/platform/barrier_test.cc"
      "${TEST_SRC_DIR}/platform/threadpool_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/controlflow/loop_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/nn/string_normalizer_test.cc"
      "${TEST_SRC_DIR}/providers/memcpy_test.cc"
    )
  endif()
endif()

set(test_all_args)
if (onnxruntime_USE_TENSORRT)
  # TRT EP CI takes much longer time when updating to TRT 8.2
  # So, we only run trt ep and exclude other eps to reduce CI test time.
  #
  # The test names of model tests were using sequential number in the past.
  # This PR https://github.com/microsoft/onnxruntime/pull/10220 (Please see ExpandModelName function in model_tests.cc for more details)
  # made test name contain the "ep" and "model path" information, so we can easily filter the tests using cuda ep or other ep with *cpu_* or *xxx_*.
  list(APPEND test_all_args "--gtest_filter=-*cpu_*:*cuda_*" )
endif ()
if(NOT onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  list(REMOVE_ITEM all_tests ${TEST_SRC_DIR}/providers/cuda/cuda_provider_test.cc)
endif()
AddTest(
  TARGET onnxruntime_test_all
  SOURCES ${all_tests} ${onnxruntime_unittest_main_src}
  LIBS
    onnx_test_runner_common ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
    onnx_test_data_proto
  DEPENDS ${all_dependencies}
  TEST_ARGS ${test_all_args}
)

if (MSVC)
  # The warning means the type of two integral values around a binary operator is narrow than their result.
  # If we promote the two input values first, it could be more tolerant to integer overflow.
  # However, this is test code. We are less concerned.
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26451>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26451>")
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")
else()
  target_compile_options(onnxruntime_test_all PRIVATE "-Wno-parentheses")
endif()

# TODO fix shorten-64-to-32 warnings
# there are some in builds where sizeof(size_t) != sizeof(int64_t), e.g., in 'ONNX Runtime Web CI Pipeline'
if (HAS_SHORTEN_64_TO_32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_options(onnxruntime_test_all PRIVATE -Wno-error=shorten-64-to-32)
endif()

if (UNIX AND onnxruntime_USE_TENSORRT)
    # The test_main.cc includes NvInfer.h where it has many deprecated declarations
    # simply ignore them for TensorRT EP build
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
# attention_op_test.cc: Function uses '49152' bytes of stack:  exceeds /analyze:stacksize '16384'..
target_compile_options(onnxruntime_test_all PRIVATE  "/analyze:stacksize 131072")
endif()

# the default logger tests conflict with the need to have an overall default logger
# so skip in this type of
target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  target_compile_definitions(onnxruntime_test_all_xc PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
endif()
if(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE)
  target_compile_definitions(onnxruntime_test_all PUBLIC -DRUN_MODELTEST_IN_DEBUG_MODE)
endif()
if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_test_all PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime_test_all PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()
if (onnxruntime_USE_ROCM)
  if (onnxruntime_USE_COMPOSABLE_KERNEL)
    target_compile_definitions(onnxruntime_test_all PRIVATE USE_COMPOSABLE_KERNEL)
  endif()
  target_compile_options(onnxruntime_test_all PRIVATE -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)
  target_include_directories(onnxruntime_test_all PRIVATE  ${onnxruntime_ROCM_HOME}/hipfft/include ${onnxruntime_ROCM_HOME}/include ${onnxruntime_ROCM_HOME}/hiprand/include ${onnxruntime_ROCM_HOME}/rocrand/include ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  target_link_libraries(onnxruntime_test_all PRIVATE Python::Python)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_DEPENDS ${TEST_SRC_DIR}/wasm/onnxruntime_test_all_adapter.js)
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_FLAGS "-s STACK_SIZE=5242880 -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4294967296 -s INCOMING_MODULE_JS_API=[preRun,locateFile,arguments,onExit,wasmMemory,buffer,instantiateWasm] --pre-js \"${TEST_SRC_DIR}/wasm/onnxruntime_test_all_adapter.js\" -s \"EXPORTED_RUNTIME_METHODS=['FS']\" --preload-file ${CMAKE_CURRENT_BINARY_DIR}/testdata@/testdata -s EXIT_RUNTIME=1 -s DEMANGLE_SUPPORT=1")
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s DEFAULT_PTHREAD_STACK_SIZE=131072 -s PROXY_TO_PTHREAD=1")
  endif()
  if (onnxruntime_USE_JSEP)
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " --pre-js \"${ONNXRUNTIME_ROOT}/wasm/js_internal_api.js\"")
  endif()

  ###
  ### if you want to investigate or debug a test failure in onnxruntime_test_all, replace the following line.
  ### those flags slow down the CI test significantly, so we don't use them by default.
  ###
  #   set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2")
  set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s SAFE_HEAP=0 -s STACK_OVERFLOW_CHECK=1")
endif()

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(onnxruntime_test_all PRIVATE ENABLE_ATEN)
endif()

set(test_data_target onnxruntime_test_all)

onnxruntime_add_static_library(onnx_test_data_proto ${TEST_SRC_DIR}/proto/tml.proto)
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
#onnx_proto target should mark this definition as public, instead of private
target_compile_definitions(onnx_test_data_proto PRIVATE "-DONNX_API=")
onnxruntime_add_include_to_target(onnx_test_data_proto onnx_proto)
target_include_directories(onnx_test_data_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(onnx_test_data_proto PROPERTIES FOLDER "ONNXRuntimeTest")
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${onnx_SOURCE_DIR} TARGET onnx_test_data_proto)

#
# onnxruntime_ir_graph test data
#
set(TEST_DATA_SRC ${TEST_SRC_DIR}/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

set(TEST_SAMPLES_SRC ${REPO_ROOT}/samples)
set(TEST_SAMPLES_DES $<TARGET_FILE_DIR:${test_data_target}>/samples)

# Copy test data from source to destination.
add_custom_command(
  TARGET ${test_data_target} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_DATA_SRC}
  ${TEST_DATA_DES})

# Copy test samples from source to destination.
add_custom_command(
  TARGET ${test_data_target} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_SAMPLES_SRC}
  ${TEST_SAMPLES_DES})

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if (onnxruntime_USE_SNPE)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${SNPE_SO_FILES} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()

  if (onnxruntime_USE_QNN)
    if (NOT QNN_ARCH_ABI)
      string(TOLOWER ${onnxruntime_target_platform} GEN_PLATFORM)
      if(MSVC)
          message(STATUS "Building MSVC for architecture ${CMAKE_SYSTEM_PROCESSOR} with CMAKE_GENERATOR_PLATFORM as ${GEN_PLATFORM}")
          if (${GEN_PLATFORM} STREQUAL "arm64")
            set(QNN_ARCH_ABI aarch64-windows-msvc)
          else()
            set(QNN_ARCH_ABI x86_64-windows-msvc)
          endif()
      else()
          if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
            set(QNN_ARCH_ABI aarch64-android-clang6.0)
          elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
            if (${GEN_PLATFORM} STREQUAL "x86_64")
              set(QNN_ARCH_ABI x86_64-linux-clang)
            else()
              set(QNN_ARCH_ABI aarch64-android)
            endif()
          endif()
      endif()
    endif()

    if (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        file(GLOB QNN_LIB_FILES LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/*.so" "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/*.dll")
        if (${QNN_ARCH_ABI} STREQUAL "aarch64-windows-msvc")
          file(GLOB EXTRA_HTP_LIB LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so" "${onnxruntime_QNN_HOME}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so")
          list(APPEND QNN_LIB_FILES ${EXTRA_HTP_LIB})
        endif()
        message(STATUS "QNN lib files: " ${QNN_LIB_FILES})
        add_custom_command(
          TARGET ${test_data_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${QNN_LIB_FILES} $<TARGET_FILE_DIR:${test_data_target}>
          )
    endif()
  endif()

  if (onnxruntime_USE_DNNL)
    if(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND onnxruntime_DNNL_OPENCL_ROOT STREQUAL "")
      message(FATAL_ERROR "--dnnl_opencl_root required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      message(FATAL_ERROR "--dnnl_gpu_runtime required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      #file(TO_CMAKE_PATH ${onnxruntime_DNNL_OPENCL_ROOT} onnxruntime_DNNL_OPENCL_ROOT)
      #set(DNNL_OCL_INCLUDE_DIR ${onnxruntime_DNNL_OPENCL_ROOT}/include)
      #set(DNNL_GPU_CMAKE_ARGS "-DDNNL_GPU_RUNTIME=OCL " "-DOPENCLROOT=${onnxruntime_DNNL_OPENCL_ROOT}")
      target_compile_definitions(onnxruntime_test_all PUBLIC -DDNNL_GPU_RUNTIME=OCL)
    endif()
    list(APPEND onnx_test_libs dnnl)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DNNL_DLL_PATH} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
  if(WIN32)
    if (onnxruntime_USE_TVM)
      add_custom_command(
        TARGET ${test_data_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tvm> $<TARGET_FILE_DIR:${test_data_target}>
        )
    endif()
  endif()

  if(WIN32)
    set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
    onnxruntime_add_static_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
    target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
    set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
    set(GETOPT_LIB_WIDE win_getopt_wide)
  endif()
endif()


set(onnx_test_libs
  onnxruntime_test_utils
  ${ONNXRUNTIME_TEST_LIBS}
  onnx_test_data_proto
  ${onnxruntime_EXTERNAL_LIBRARIES})

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnx_test_libs onnxruntime_language_interop onnxruntime_pyop)
endif()

onnxruntime_add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
if(MSVC)
  target_compile_options(onnx_test_runner PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
  set_target_properties(onnx_test_runner PROPERTIES
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
  )
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
  else()
    set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1")
  endif()
endif()

target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs} nlohmann_json::nlohmann_json)
target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})
if (onnxruntime_USE_ROCM)
  target_include_directories(onnx_test_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  target_link_libraries(onnx_test_runner PRIVATE Python::Python)
endif()
set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

if (onnxruntime_USE_TVM)
  if (WIN32)
    target_link_options(onnx_test_runner PRIVATE "/STACK:4000000")
  endif()
endif()

install(TARGETS onnx_test_runner
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        BUNDLE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if(onnxruntime_BUILD_BENCHMARKS)
    SET(BENCHMARK_DIR ${TEST_SRC_DIR}/onnx/microbenchmark)
    onnxruntime_add_executable(onnxruntime_benchmark
      ${BENCHMARK_DIR}/main.cc
      ${BENCHMARK_DIR}/modeltest.cc
      ${BENCHMARK_DIR}/pooling.cc
      ${BENCHMARK_DIR}/resize.cc
      ${BENCHMARK_DIR}/batchnorm.cc
      ${BENCHMARK_DIR}/batchnorm2.cc
      ${BENCHMARK_DIR}/tptest.cc
      ${BENCHMARK_DIR}/eigen.cc
      ${BENCHMARK_DIR}/copy.cc
      ${BENCHMARK_DIR}/gelu.cc
      ${BENCHMARK_DIR}/activation.cc
      ${BENCHMARK_DIR}/quantize.cc
      ${BENCHMARK_DIR}/reduceminmax.cc)
    target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_compile_definitions(onnxruntime_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    if(WIN32)
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4141>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4141>")
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26400>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26400>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26814>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26497>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_link_libraries(onnxruntime_benchmark PRIVATE onnx_test_runner_common benchmark::benchmark ${onnx_test_libs})
    add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")

    SET(MLAS_BENCH_DIR ${TEST_SRC_DIR}/mlas/bench)
    file(GLOB_RECURSE MLAS_BENCH_SOURCE_FILES "${MLAS_BENCH_DIR}/*.cpp" "${MLAS_BENCH_DIR}/*.h")
    onnxruntime_add_executable(onnxruntime_mlas_benchmark ${MLAS_BENCH_SOURCE_FILES})
    target_include_directories(onnxruntime_mlas_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(onnxruntime_mlas_benchmark PRIVATE benchmark::benchmark onnxruntime_util onnxruntime_framework ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE debug Dbghelp)
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26409)
      # "Global initializer calls a non-constexpr function." BENCHMARK_CAPTURE macro needs this.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26426)
    else()
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE nsync::nsync_cpp ${CMAKE_DL_LIBS})
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE cpuinfo)
    endif()
    set_target_properties(onnxruntime_mlas_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  if(WIN32)
    target_compile_options(onnx_test_runner_common PRIVATE -D_CRT_SECURE_NO_WARNINGS)
  endif()

  if (NOT onnxruntime_REDUCED_OPS_BUILD AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    add_test(NAME onnx_test_pytorch_converted
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-converted)
    add_test(NAME onnx_test_pytorch_operator
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-operator)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND android_shared_libs log android)
  endif()
endif()


if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  #perf test runner
  set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
  set(onnxruntime_perf_test_src_patterns
  "${onnxruntime_perf_test_src_dir}/*.cc"
  "${onnxruntime_perf_test_src_dir}/*.h")

  if(WIN32)
    list(APPEND onnxruntime_perf_test_src_patterns
      "${onnxruntime_perf_test_src_dir}/windows/*.cc"
      "${onnxruntime_perf_test_src_dir}/windows/*.h" )
  else ()
    list(APPEND onnxruntime_perf_test_src_patterns
      "${onnxruntime_perf_test_src_dir}/posix/*.cc"
      "${onnxruntime_perf_test_src_dir}/posix/*.h" )
  endif()

  file(GLOB onnxruntime_perf_test_src CONFIGURE_DEPENDS
    ${onnxruntime_perf_test_src_patterns}
    )
  onnxruntime_add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src} ${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc)
  if(MSVC)
    target_compile_options(onnxruntime_perf_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()
  target_include_directories(onnxruntime_perf_test PRIVATE ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
          ${eigen_INCLUDE_DIRS} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
          ${CMAKE_CURRENT_BINARY_DIR})
  if (onnxruntime_USE_ROCM)
    target_include_directories(onnxruntime_perf_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
  endif()
  if (WIN32)
    target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
    if (NOT DEFINED SYS_PATH_LIB)
      set(SYS_PATH_LIB shlwapi)
    endif()
  endif()
  if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnxruntime_perf_test PROPERTIES
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    )
  endif()

  if (onnxruntime_BUILD_SHARED_LIB)
    #It will dynamically link to onnxruntime. So please don't add onxruntime_graph/onxruntime_framework/... here.
    #onnxruntime_common is kind of ok because it is thin, tiny and totally stateless.
    set(onnxruntime_perf_test_libs
            onnx_test_runner_common onnxruntime_test_utils onnxruntime_common
            onnxruntime onnxruntime_flatbuffers onnx_test_data_proto
            ${onnxruntime_EXTERNAL_LIBRARIES}
            ${GETOPT_LIB_WIDE} ${SYS_PATH_LIB} ${CMAKE_DL_LIBS})
    if(NOT WIN32)
      list(APPEND onnxruntime_perf_test_libs nsync::nsync_cpp)
      if(onnxruntime_USE_SNPE)
        list(APPEND onnxruntime_perf_test_libs onnxruntime_providers_snpe)
      endif()
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_perf_test_libs ${android_shared_libs})
    endif()
    target_link_libraries(onnxruntime_perf_test PRIVATE ${onnxruntime_perf_test_libs} Threads::Threads)
    if(WIN32)
      target_link_libraries(onnxruntime_perf_test PRIVATE debug dbghelp advapi32)
    endif()
    if(tensorflow_C_PACKAGE_PATH)
      target_include_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/include)
      target_link_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/lib)
      target_link_libraries(onnxruntime_perf_test PRIVATE tensorflow)
      target_compile_definitions(onnxruntime_perf_test PRIVATE HAVE_TENSORFLOW)
    endif()
  else()
    target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
  endif()
  set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")

  if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS AND NOT onnxruntime_BUILD_SHARED_LIB)
    target_link_libraries(onnxruntime_perf_test PRIVATE onnxruntime_language_interop onnxruntime_pyop)
  endif()

  if (onnxruntime_USE_TVM)
    if (WIN32)
      target_link_options(onnxruntime_perf_test PRIVATE "/STACK:4000000")
    endif()
  endif()

  # shared lib
  if (onnxruntime_BUILD_SHARED_LIB)
    onnxruntime_add_static_library(onnxruntime_mocked_allocator ${TEST_SRC_DIR}/util/test_allocator.cc)
    target_include_directories(onnxruntime_mocked_allocator PUBLIC ${TEST_SRC_DIR}/util/include)
    target_link_libraries(onnxruntime_mocked_allocator PRIVATE ${GSL_TARGET})
    set_target_properties(onnxruntime_mocked_allocator PROPERTIES FOLDER "ONNXRuntimeTest")

    #################################################################
    # test inference using shared lib
    set(onnxruntime_shared_lib_test_LIBS onnxruntime_mocked_allocator onnxruntime_test_utils onnxruntime_common onnx_proto)
    if(NOT WIN32)
      list(APPEND onnxruntime_shared_lib_test_LIBS nsync::nsync_cpp)
      if(onnxruntime_USE_SNPE)
        list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_providers_snpe)
      endif()
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_shared_lib_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_CUDA)
      list(APPEND onnxruntime_shared_lib_test_LIBS cudart)
    endif()
    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_shared_lib_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_shared_lib_test_LIBS ${android_shared_libs})
    endif()

    AddTest(DYN
            TARGET onnxruntime_shared_lib_test
            SOURCES ${onnxruntime_shared_lib_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_shared_lib_test_LIBS}
            DEPENDS ${all_dependencies}
    )
    if (onnxruntime_USE_CUDA)
      target_include_directories(onnxruntime_shared_lib_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      target_sources(onnxruntime_shared_lib_test PRIVATE ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_sources(onnxruntime_shared_lib_test PRIVATE
        "${ONNXRUNTIME_ROOT}/core/platform/android/cxa_demangle.cc"
        "${TEST_SRC_DIR}/platform/android/cxa_demangle_test.cc"
      )
      target_compile_definitions(onnxruntime_shared_lib_test PRIVATE USE_DUMMY_EXA_DEMANGLE=1)
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
      add_custom_command(
        TARGET onnxruntime_shared_lib_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
        $<TARGET_FILE_DIR:onnxruntime_shared_lib_test>/testdata)
    endif()

    if (UNIX AND onnxruntime_USE_TENSORRT)
        # The test_main.cc includes NvInfer.h where it has many deprecated declarations
        # simply ignore them for TensorRT EP build
        set_property(TARGET onnxruntime_shared_lib_test APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    endif()

    # test inference using global threadpools
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Android|iOS" AND NOT onnxruntime_MINIMAL_BUILD)
      AddTest(DYN
              TARGET onnxruntime_global_thread_pools_test
              SOURCES ${onnxruntime_global_thread_pools_test_SRC}
              LIBS ${onnxruntime_shared_lib_test_LIBS}
              DEPENDS ${all_dependencies}
      )
    endif()
  endif()

  # the debug node IO functionality uses static variables, so it is best tested
  # in its own process
  if(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
    AddTest(
      TARGET onnxruntime_test_debug_node_inputs_outputs
      SOURCES
        "${TEST_SRC_DIR}/debug_node_inputs_outputs/debug_node_inputs_outputs_utils_test.cc"
        "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
        "${TEST_SRC_DIR}/framework/test_utils.cc"
        "${TEST_SRC_DIR}/providers/base_tester.h"
        "${TEST_SRC_DIR}/providers/base_tester.cc"
        "${TEST_SRC_DIR}/providers/checkers.h"
        "${TEST_SRC_DIR}/providers/checkers.cc"
        "${TEST_SRC_DIR}/providers/op_tester.h"
        "${TEST_SRC_DIR}/providers/op_tester.cc"
        "${TEST_SRC_DIR}/providers/provider_test_utils.h"
        "${TEST_SRC_DIR}/providers/tester_types.h"
        ${onnxruntime_unittest_main_src}
      LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
      DEPENDS ${all_dependencies}
    )

    if (onnxruntime_USE_ROCM)
      target_include_directories(onnxruntime_test_debug_node_inputs_outputs PRIVATE ${onnxruntime_ROCM_HOME}/hipfft/include ${onnxruntime_ROCM_HOME}/include ${onnxruntime_ROCM_HOME}/hipcub/include ${onnxruntime_ROCM_HOME}/hiprand/include ${onnxruntime_ROCM_HOME}/rocrand/include)
      target_include_directories(onnxruntime_test_debug_node_inputs_outputs PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime)
    endif(onnxruntime_USE_ROCM)

    target_compile_definitions(onnxruntime_test_debug_node_inputs_outputs
      PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
  endif(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)

  #some ETW tools
  if(WIN32 AND onnxruntime_ENABLE_INSTRUMENT)
    onnxruntime_add_executable(generate_perf_report_from_etl ${ONNXRUNTIME_ROOT}/tool/etw/main.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(generate_perf_report_from_etl PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(generate_perf_report_from_etl PRIVATE tdh Advapi32)

    onnxruntime_add_executable(compare_two_sessions ${ONNXRUNTIME_ROOT}/tool/etw/compare_two_sessions.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(compare_two_sessions PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(compare_two_sessions PRIVATE ${GETOPT_LIB_WIDE} tdh Advapi32)
  endif()

  if(NOT onnxruntime_target_platform STREQUAL "ARM64EC")
    file(GLOB onnxruntime_mlas_test_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/mlas/unittest/*.h"
      "${TEST_SRC_DIR}/mlas/unittest/*.cpp"
    )
    onnxruntime_add_executable(onnxruntime_mlas_test ${onnxruntime_mlas_test_src})
    if(MSVC)
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
    endif()
    if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
      set_target_properties(onnxruntime_mlas_test PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
      )
    endif()
    target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnxruntime_mlas_test PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_test PRIVATE cpuinfo)
    endif()
    if(NOT WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE nsync::nsync_cpp ${CMAKE_DL_LIBS})
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(onnxruntime_mlas_test PRIVATE ${android_shared_libs})
    endif()
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE debug Dbghelp Advapi32)
    endif()
    if (onnxruntime_LINK_LIBATOMIC)
      target_link_libraries(onnxruntime_mlas_test PRIVATE atomic)
    endif()
    target_link_libraries(onnxruntime_mlas_test PRIVATE Threads::Threads)
    set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
      else()
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
      endif()
    endif()
endif()
  # Training API Tests
  # Disabling training_api_test_trainer. CXXOPT generates a ton of warnings because of which nuget pipeline is failing.
  # TODO(askhade): Fix the warnings.
  # This has no impact on the release as the release package and the pipeline, both do not use this.
  # This is used by devs for testing training apis.
  #if (onnxruntime_ENABLE_TRAINING_APIS)
  if (0)
    # Only files in the trainer and common folder will be compiled into test trainer.
    file(GLOB training_api_test_trainer_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.h"
    )
    onnxruntime_add_executable(onnxruntime_test_trainer ${training_api_test_trainer_src})

    onnxruntime_add_include_to_target(onnxruntime_test_trainer onnxruntime_session
      onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)

    set(CXXOPTS ${cxxopts_SOURCE_DIR}/include)
    target_include_directories(onnxruntime_test_trainer PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${ONNXRUNTIME_ROOT}
      ${ORTTRAINING_ROOT}
      ${eigen_INCLUDE_DIRS}
      ${CXXOPTS}
      ${extra_includes}
      ${onnxruntime_graph_header}
      ${onnxruntime_exec_src_dir}
    )

    set(ONNXRUNTIME_TEST_LIBS
      onnxruntime_session
      ${onnxruntime_libs}
      # CUDA is dynamically loaded at runtime
      onnxruntime_optimizer
      onnxruntime_providers
      onnxruntime_util
      onnxruntime_framework
      onnxruntime_util
      onnxruntime_graph
      ${ONNXRUNTIME_MLAS_LIBS}
      onnxruntime_common
      onnxruntime_flatbuffers
    )

    if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
      list(APPEND ONNXRUNTIME_TEST_LIBS onnxruntime_language_interop onnxruntime_pyop)
    endif()

    target_link_libraries(onnxruntime_test_trainer PRIVATE
      ${ONNXRUNTIME_TEST_LIBS}
      ${onnxruntime_EXTERNAL_LIBRARIES}
    )
    set_target_properties(onnxruntime_test_trainer PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")

  set(custom_op_src_patterns
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.h"
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.cc"
    "${TEST_SRC_DIR}/testdata/custom_op_library/cpu/cpu_ops.*"
  )

  set(custom_op_lib_include ${REPO_ROOT}/include)
  set(custom_op_lib_option)
  set(custom_op_lib_link ${GSL_TARGET})

  if (onnxruntime_USE_CUDA)
    list(APPEND custom_op_src_patterns
        "${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu"
        "${TEST_SRC_DIR}/testdata/custom_op_library/cuda/cuda_ops.*")
    list(APPEND custom_op_lib_include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${onnxruntime_CUDNN_HOME}/include)
    if (HAS_QSPECTRE)
      list(APPEND custom_op_lib_option "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /Qspectre>")
    endif()
  endif()

  if (onnxruntime_USE_ROCM)
    list(APPEND custom_op_src_patterns
        "${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/rocm_ops.hip"
        "${TEST_SRC_DIR}/testdata/custom_op_library/rocm/rocm_ops.*")
    list(APPEND custom_op_lib_include ${onnxruntime_ROCM_HOME}/include)
    list(APPEND custom_op_lib_option "-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1")
  endif()

  file(GLOB custom_op_src ${custom_op_src_patterns})
  onnxruntime_add_shared_library(custom_op_library ${custom_op_src})
  target_compile_options(custom_op_library PRIVATE ${custom_op_lib_option})
  target_include_directories(custom_op_library PRIVATE ${REPO_ROOT}/include ${custom_op_lib_include})
  target_link_libraries(custom_op_library PRIVATE ${GSL_TARGET} ${custom_op_lib_link})

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.def")
    if (NOT onnxruntime_USE_CUDA)
      target_compile_options(custom_op_library PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
    endif()
  endif()
  set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})

  if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    if (onnxruntime_BUILD_JAVA AND NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
        message(STATUS "Running Java tests")
        # native-test is added to resources so custom_op_lib can be loaded
        # and we want to symlink it there
        set(JAVA_NATIVE_TEST_DIR ${JAVA_OUTPUT_DIR}/native-test)
        file(MAKE_DIRECTORY ${JAVA_NATIVE_TEST_DIR})

        # delegate to gradle's test runner
        if(WIN32)
          add_custom_command(TARGET custom_op_library POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:custom_op_library>
                          ${JAVA_NATIVE_TEST_DIR}/$<TARGET_FILE_NAME:custom_op_library>)
          # On windows ctest requires a test to be an .exe(.com) file
          # With gradle wrapper we get gradlew.bat. We delegate execution to a separate .cmake file
          # That can handle both .exe and .bat
          add_test(NAME onnxruntime4j_test COMMAND ${CMAKE_COMMAND}
            -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE}
            -DBIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DREPO_ROOT=${REPO_ROOT}
            ${ORT_PROVIDER_FLAGS}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_java_unittests.cmake)
        else()
          add_custom_command(TARGET custom_op_library POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:custom_op_library>
                          ${JAVA_NATIVE_TEST_DIR}/$<TARGET_LINKER_FILE_NAME:custom_op_library>)
          if (onnxruntime_ENABLE_TRAINING_APIS)
            message(STATUS "Running Java inference and training tests")
            add_test(NAME onnxruntime4j_test COMMAND ${GRADLE_EXECUTABLE} cmakeCheck -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} ${ORT_PROVIDER_FLAGS} -DENABLE_TRAINING_APIS=1
                          WORKING_DIRECTORY ${REPO_ROOT}/java)
          else()
            message(STATUS "Running Java inference tests only")
            add_test(NAME onnxruntime4j_test COMMAND ${GRADLE_EXECUTABLE} cmakeCheck -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} ${ORT_PROVIDER_FLAGS}
                          WORKING_DIRECTORY ${REPO_ROOT}/java)
          endif()
        endif()
        set_property(TEST onnxruntime4j_test APPEND PROPERTY DEPENDS onnxruntime4j_jni)
    endif()
  endif()

  if (onnxruntime_BUILD_SHARED_LIB AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
    set (onnxruntime_customopregistration_test_SRC
            ${ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR}/test_registercustomops.cc)

    set(onnxruntime_customopregistration_test_LIBS custom_op_library onnxruntime_common onnxruntime_test_utils)
    if (NOT WIN32)
      list(APPEND onnxruntime_customopregistration_test_LIBS nsync::nsync_cpp)
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_customopregistration_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_customopregistration_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    AddTest(DYN
            TARGET onnxruntime_customopregistration_test
            SOURCES ${onnxruntime_customopregistration_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_customopregistration_test_LIBS}
            DEPENDS ${all_dependencies}
    )

    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
      add_custom_command(
        TARGET onnxruntime_customopregistration_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
        $<TARGET_FILE_DIR:onnxruntime_customopregistration_test>/testdata)
    endif()

    if (UNIX AND onnxruntime_USE_TENSORRT)
        # The test_main.cc includes NvInfer.h where it has many deprecated declarations
        # simply ignore them for TensorRT EP build
        set_property(TARGET onnxruntime_customopregistration_test APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    endif()

  endif()
endif()

# Build custom op library that returns an error OrtStatus when the exported RegisterCustomOps function is called.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_invalid_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.h
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.cc)
  target_include_directories(custom_op_invalid_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.def")
  endif()

  set_property(TARGET custom_op_invalid_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG})
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))

  file(GLOB_RECURSE custom_op_get_const_input_test_library_src
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.h"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.cc"
  )

  onnxruntime_add_shared_library_module(custom_op_get_const_input_test_library ${custom_op_get_const_input_test_library_src})

  onnxruntime_add_include_to_target(custom_op_get_const_input_test_library onnxruntime_common GTest::gtest GTest::gmock)
  target_include_directories(custom_op_get_const_input_test_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session
                                                                            ${REPO_ROOT}/include/onnxruntime/core/common)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_get_const_input_test_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG})
endif()

if (onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  set (onnxruntime_logging_apis_test_SRC
       ${ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR}/test_logging_apis.cc)

  set(onnxruntime_logging_apis_test_LIBS onnxruntime_common onnxruntime_test_utils)

  if(NOT WIN32)
    list(APPEND onnxruntime_logging_apis_test_LIBS nsync::nsync_cpp ${CMAKE_DL_LIBS})
  endif()

  AddTest(DYN
          TARGET onnxruntime_logging_apis_test
          SOURCES ${onnxruntime_logging_apis_test_SRC}
          LIBS ${onnxruntime_logging_apis_test_LIBS}
          DEPENDS ${all_dependencies}
  )
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_USE_OPENVINO AND (NOT onnxruntime_MINIMAL_BUILD OR
                                                                        onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_openvino_wrapper_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.cc
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/openvino_wrapper.cc)
  target_include_directories(custom_op_openvino_wrapper_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(custom_op_openvino_wrapper_library PRIVATE openvino::runtime)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_openvino_wrapper_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG})
endif()

# limit to only test on windows first, due to a runtime path issue on linux
if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten"
                                  AND NOT onnxruntime_USE_ROCM)
  file(GLOB_RECURSE test_execution_provider_srcs
    "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.h"
    "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  onnxruntime_add_shared_library_module(test_execution_provider ${test_execution_provider_srcs})
  add_dependencies(test_execution_provider onnxruntime_providers_shared onnx ${ABSEIL_LIBS})
  target_link_libraries(test_execution_provider PRIVATE onnxruntime_providers_shared ${ABSEIL_LIBS} Boost::mp11)
  target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnxruntime_common,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(test_execution_provider PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${ORTTRAINING_ROOT})
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    target_link_libraries(test_execution_provider PRIVATE Python::Python)
  endif()
  if(APPLE)
    set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\\$ORIGIN")
  elseif(WIN32)
    set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/symbols.def")
  else()
    message(FATAL_ERROR "test_execution_provider unknown platform, need to specify shared library exports for it")
  endif()
endif()

include(onnxruntime_fuzz_test.cmake)
