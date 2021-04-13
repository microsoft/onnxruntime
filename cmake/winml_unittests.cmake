# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(WINML_TEST_SRC_DIR ${REPO_ROOT}/winml/test)
set(WINML_TEST_INC_DIR
  ${REPO_ROOT}/winml/api
  ${REPO_ROOT}/winml/test/common
  ${REPO_ROOT}/winml/lib/Common/inc
  ${REPO_ROOT}/onnxruntime
  ${REPO_ROOT}/onnxruntime/core/providers/dml/DmlExecutionProvider/src/External/D3DX12
  ${REPO_ROOT}/cmake/external/googletest/googletest/include
  ${REPO_ROOT}/cmake/external/protobuf/src
  ${REPO_ROOT}/cmake/external/wil/include
  ${REPO_ROOT}/cmake/external/SafeInt
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated
  ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api_experimental
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api_experimental/comp_generated
)

function(set_winml_target_properties target)
  set_target_properties(${target} PROPERTIES
    FOLDER "ONNXRuntimeTest/winml"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
  )
  target_include_directories(${target} PRIVATE ${WINML_TEST_INC_DIR})
  target_compile_definitions(${target} PRIVATE WINML_ROOT_NS=${winml_root_ns})
  target_compile_definitions(${target} PRIVATE BINARY_NAME=\"${BINARY_NAME}\")
endfunction()

function(add_winml_test)
  # Add a test target and make it discoverable by CTest by calling add_test
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})
  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()
  list(REMOVE_DUPLICATES _UT_SOURCES)
  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif()

  onnxruntime_add_executable(${_UT_TARGET} ${_UT_SOURCES})
  onnxruntime_add_include_to_target(${_UT_TARGET} onnx_proto)
  source_group(TREE ${WINML_TEST_SRC_DIR} FILES ${_UT_SOURCES})
  set_winml_target_properties(${_UT_TARGET})
  target_compile_definitions(${_UT_TARGET} PRIVATE BUILD_GOOGLE_TEST)
  target_precompiled_header(${_UT_TARGET} testPch.h)

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif()
  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest winml_google_test_lib ${onnxruntime_EXTERNAL_LIBRARIES} winml_lib_common onnxruntime windowsapp.lib)
  target_compile_options(${_UT_TARGET} PRIVATE /wd5205)  # workaround cppwinrt SDK bug https://github.com/microsoft/cppwinrt/issues/584

  # if building inbox
  if (onnxruntime_WINML_NAMESPACE_OVERRIDE STREQUAL "Windows")
    target_compile_definitions(${_UT_TARGET} PRIVATE "BUILD_INBOX=1")
  endif()

  if (onnxruntime_BUILD_MS_EXPERIMENTAL_OPS)
    target_compile_definitions(${_UT_TARGET} PRIVATE "BUILD_MS_EXPERIMENTAL_OPS=1")
  endif()

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
  )
endfunction()

function(get_winml_test_scenario_src
  winml_test_src_path
  output_winml_test_scenario_src
  output_winml_test_scenario_libs
)
  if (onnxruntime_USE_DML)
    file(GLOB winml_test_scenario_src CONFIGURE_DEPENDS
        "${winml_test_src_path}/scenario/cppwinrt/*.h"
        "${winml_test_src_path}/scenario/cppwinrt/*.cpp")
    set(${output_winml_test_scenario_libs} "onnxruntime_providers_dml" PARENT_SCOPE)
  else()
    set(winml_test_scenario_src
        "${winml_test_src_path}/scenario/cppwinrt/scenariotestscppwinrt.h"
        "${winml_test_src_path}/scenario/cppwinrt/scenariotestscppwinrt.cpp"
        )
  endif()
  set(${output_winml_test_scenario_src} ${winml_test_scenario_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_api_src
  winml_test_src_path
  output_winml_test_api_src
)
  file(GLOB winml_test_api_src CONFIGURE_DEPENDS
      "${winml_test_src_path}/api/APITest.h"
      "${winml_test_src_path}/api/LearningModelAPITest.h"
      "${winml_test_src_path}/api/LearningModelBindingAPITest.h"
      "${winml_test_src_path}/api/LearningModelSessionAPITest.h"
      "${winml_test_src_path}/api/LearningModelAPITest.cpp"
      "${winml_test_src_path}/api/LearningModelBindingAPITest.cpp"
      "${winml_test_src_path}/api/LearningModelSessionAPITest.cpp")

  set(${output_winml_test_api_src} ${winml_test_api_src} ${winml_redist_only_api_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_api_redist_only_src
  winml_test_src_path
  output_winml_test_api_src
)
  file(GLOB winml_redist_only_api_src CONFIGURE_DEPENDS
  "${winml_test_src_path}/api/RawApiHelpers.h"
  "${winml_test_src_path}/api/RawApiTests.h"
  "${winml_test_src_path}/api/RawApiTestsGpu.h"
  "${winml_test_src_path}/api/RawApiHelpers.cpp"
  "${winml_test_src_path}/api/RawApiTests.cpp"
  "${winml_test_src_path}/api/RawApiTestsGpu.cpp"
  "${winml_test_src_path}/api/raw/*.h"
  "${winml_test_src_path}/api/raw/*.cpp")

  set(${output_winml_test_api_src} ${winml_test_api_src} ${winml_redist_only_api_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_concurrency_src
  winml_test_src_path
  output_winml_test_concurrency_src
)
  file(GLOB winml_test_concurrency_src CONFIGURE_DEPENDS
      "${winml_test_src_path}/concurrency/*.h"
      "${winml_test_src_path}/concurrency/*.cpp")
  set(${output_winml_test_concurrency_src} ${winml_test_concurrency_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_adapter_src
  winml_test_src_path
  output_winml_test_adapter_src
  output_winml_test_adapter_libs
)
  set(${output_winml_test_adapter_libs} onnxruntime winml_lib_ort winml_test_common PARENT_SCOPE)
  file(GLOB winml_test_adapter_src CONFIGURE_DEPENDS
      "${winml_test_src_path}/adapter/*.h"
      "${winml_test_src_path}/adapter/*.cpp")
  set(${output_winml_test_adapter_src} ${winml_test_adapter_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_image_src
  winml_test_src_path
  output_winml_test_image_src
)
  if (onnxruntime_USE_DML)
    set(${output_winml_test_scenario_libs} "onnxruntime_providers_dml" PARENT_SCOPE)
  endif()
  file(GLOB winml_test_image_src CONFIGURE_DEPENDS
      "${winml_test_src_path}/image/*.h"
      "${winml_test_src_path}/image/*.cpp")
  set(${output_winml_test_image_src} ${winml_test_image_src} PARENT_SCOPE)
endfunction()

function (get_winml_test_model_src
  winml_test_src_path
  output_winml_test_model_src
  winml_test_model_libs)
  file(GLOB winml_test_model_src CONFIGURE_DEPENDS
      "${winml_test_src_path}/model/*.h"
      "${winml_test_src_path}/model/*.cpp")
  set(${output_winml_test_model_src} ${winml_test_model_src} PARENT_SCOPE)
  set(${winml_test_model_libs} onnx_test_data_proto onnx_test_runner_common onnxruntime_common onnxruntime_mlas
    onnxruntime_graph onnxruntime_test_utils onnxruntime_framework onnxruntime_util onnxruntime_flatbuffers PARENT_SCOPE)
endfunction()

file(GLOB winml_test_common_src CONFIGURE_DEPENDS
    "${WINML_TEST_SRC_DIR}/common/*.h"
    "${WINML_TEST_SRC_DIR}/common/*.cpp")
onnxruntime_add_static_library(winml_test_common STATIC ${winml_test_common_src})
target_compile_options(winml_test_common PRIVATE /wd5205)  # workaround cppwinrt SDK bug https://github.com/microsoft/cppwinrt/issues/584
if (onnxruntime_WINML_NAMESPACE_OVERRIDE STREQUAL "Windows")
  target_compile_definitions(winml_test_common PRIVATE "BUILD_INBOX=1")
endif()
add_dependencies(winml_test_common
  onnx
  winml_api
  winml_dll
)
onnxruntime_add_include_to_target(winml_test_common onnx_proto)
onnxruntime_add_static_library(winml_google_test_lib STATIC ${WINML_TEST_SRC_DIR}/common/googletest/main.cpp)
set_winml_target_properties(winml_google_test_lib)

set_winml_target_properties(winml_test_common)
get_winml_test_api_src(${WINML_TEST_SRC_DIR} winml_test_api_src)

if (NOT ${winml_is_inbox})
  get_winml_test_api_redist_only_src(${WINML_TEST_SRC_DIR} winml_test_api_redist_only_src)
endif()

add_winml_test(
  TARGET winml_test_api
  SOURCES ${winml_test_api_src} ${winml_test_api_redist_only_src}
  LIBS winml_test_common
)
target_delayload(winml_test_api dxgi.dll d3d12.dll api-ms-win-core-file-l1-2-2.dll api-ms-win-core-synch-l1-2-1.dll)
if (onnxruntime_USE_DML)
  target_delayload(winml_test_api directml.dll)
endif()
if (EXISTS ${dxcore_header})
  target_delayload(winml_test_api ext-ms-win-dxcore-l1-*.dll)
endif()

get_winml_test_scenario_src(${WINML_TEST_SRC_DIR} winml_test_scenario_src winml_test_scenario_libs)
add_winml_test(
  TARGET winml_test_scenario
  SOURCES ${winml_test_scenario_src}
  LIBS winml_test_common ${winml_test_scenario_libs}
)
target_delayload(winml_test_scenario d2d1.dll d3d11.dll dxgi.dll d3d12.dll api-ms-win-core-libraryloader-l1-2-1.dll api-ms-win-core-file-l1-2-2.dll api-ms-win-core-synch-l1-2-1.dll)
if (onnxruntime_USE_DML)
  target_delayload(winml_test_scenario directml.dll)
endif()
if (EXISTS ${dxcore_header})
  target_delayload(winml_test_scenario ext-ms-win-dxcore-l1-*.dll)
endif()

# necessary for winml_test_scenario because of a still unknown reason, api-ms-win-core-libraryloader-l1-2-1.dll is linked against
# on dev machines but not on the aiinfra agent pool
target_link_options(winml_test_scenario PRIVATE /ignore:4199)

get_winml_test_image_src(${WINML_TEST_SRC_DIR} winml_test_image_src winml_test_image_libs)
add_winml_test(
  TARGET winml_test_image
  SOURCES ${winml_test_image_src}
  LIBS winml_test_common ${winml_test_image_libs}
)
target_precompiled_header(winml_test_image testPch.h)
if(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE)
  target_compile_definitions(winml_test_image PUBLIC -DRUN_MODELTEST_IN_DEBUG_MODE)
endif()
target_delayload(winml_test_image d3d12.dll api-ms-win-core-file-l1-2-2.dll api-ms-win-core-synch-l1-2-1.dll)

get_winml_test_concurrency_src(${WINML_TEST_SRC_DIR} winml_test_concurrency_src)
add_winml_test(
  TARGET winml_test_concurrency
  SOURCES ${winml_test_concurrency_src}
  LIBS winml_test_common
)
target_include_directories(winml_test_concurrency PRIVATE ${ONNXRUNTIME_ROOT}/core/graph)
target_include_directories(winml_test_concurrency PRIVATE ${ONNXRUNTIME_ROOT}/winml/lib/Api.Ort)

get_winml_test_adapter_src(${WINML_TEST_SRC_DIR} winml_test_adapter_src winml_test_adapter_libs)
add_winml_test(
  TARGET winml_test_adapter
  SOURCES ${winml_test_adapter_src}
  LIBS ${winml_test_adapter_libs}
)
target_include_directories(winml_test_adapter PRIVATE ${REPO_ROOT}/winml/adapter)
target_include_directories(winml_test_adapter PRIVATE ${REPO_ROOT}/winml/lib/Api.Ort)

target_include_directories(winml_test_adapter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                   # windows machine learning generated component headers
target_include_directories(winml_test_adapter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)    # windows machine learning generated component headers
target_include_directories(winml_test_adapter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers

target_include_directories(winml_test_adapter PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(winml_test_adapter PRIVATE ${REPO_ROOT}/winml ${REPO_ROOT}/winml/lib/Api/inc)
target_include_directories(winml_test_adapter PRIVATE ${winml_lib_api_dir})                            # needed for generated headers
target_include_directories(winml_test_adapter PRIVATE ${winml_lib_api_core_dir})
target_include_directories(winml_test_adapter PRIVATE ${winml_lib_api_ort_dir})
target_include_directories(winml_test_adapter PRIVATE ${winml_lib_common_dir}/inc)
target_include_directories(winml_test_adapter PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_include_directories(winml_test_adapter PRIVATE ${ONNXRUNTIME_ROOT})

onnxruntime_add_include_to_target(winml_test_adapter onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
target_include_directories(winml_test_adapter PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(winml_test_adapter ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(winml_test_adapter PRIVATE ${winml_adapter_dir})
target_include_directories(winml_test_adapter PRIVATE ${winml_lib_common_dir}/inc)

# Onnxruntime memory leak checker doesn't work well with GTest static mutexes that create critical sections that cannot be freed prematurely.
if(NOT onnxruntime_ENABLE_MEMLEAK_CHECKER)
  get_winml_test_model_src(${WINML_TEST_SRC_DIR} winml_test_model_src winml_test_model_libs)
  add_winml_test(
    TARGET winml_test_model
    SOURCES ${winml_test_model_src}
    LIBS winml_test_common ${winml_test_model_libs}
  )
  if (EXISTS ${dxcore_header})
    target_delayload(winml_test_model ext-ms-win-dxcore-l1-*.dll)
  endif()
  target_precompiled_header(winml_test_model testPch.h)
endif()

# During build time, copy any modified collaterals.
# configure_file(source destination COPYONLY), which configures CMake to copy the file whenever source is modified,
# can't be used here because we don't know the destination during configure time (in multi-configuration generators,
# such as VS, one can switch between Debug/Release builds in the same build tree, and the destination depends on the
# build mode).
function(add_winml_collateral source)
  get_filename_component(source_directory ${source} DIRECTORY)
  file(GLOB_RECURSE collaterals RELATIVE ${source_directory} ${source})
  foreach(collateral ${collaterals})
    set(collateral_path ${source_directory}/${collateral})
    if(NOT IS_DIRECTORY ${collateral_path})
        add_custom_command(TARGET winml_test_common
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${collateral_path} "$<TARGET_FILE_DIR:winml_test_common>/${collateral}")
    endif()
  endforeach()
endfunction()

add_winml_collateral("${WINML_TEST_SRC_DIR}/api/models/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/collateral/images/*.jpg")
add_winml_collateral("${WINML_TEST_SRC_DIR}/collateral/images/*.png")
add_winml_collateral("${WINML_TEST_SRC_DIR}/collateral/models/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/common/testdata/squeezenet/*")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/images/*.jpg")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/images/*.png")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/groundTruth/*.jpg")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/groundTruth/*.png")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/batchGroundTruth/*.jpg")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/batchGroundTruth/*.png")
add_winml_collateral("${WINML_TEST_SRC_DIR}/image/models/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/scenario/cppwinrt/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/scenario/models/*.onnx")
add_winml_collateral("${REPO_ROOT}/onnxruntime/test/testdata/sequence_length.onnx")
add_winml_collateral("${REPO_ROOT}/onnxruntime/test/testdata/sequence_construct.onnx")
