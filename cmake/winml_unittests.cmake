# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(WINML_TEST_SRC_DIR ${REPO_ROOT}/winml/test)
set(WINML_TEST_INC_DIR
  ${REPO_ROOT}/winml/test/common
  ${REPO_ROOT}/winml/lib/Common/inc
  ${REPO_ROOT}/onnxruntime
  ${REPO_ROOT}/onnxruntime/core/providers/dml/DmlExecutionProvider/src/External/D3DX12
  ${REPO_ROOT}/cmake/external/googletest/googletest/include
  ${REPO_ROOT}/cmake/external/protobuf/src
  ${REPO_ROOT}/cmake/external/wil/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated
  ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)

function(set_winml_target_properties target)
  set_target_properties(${target} PROPERTIES
    FOLDER "ONNXRuntimeTest/winml"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
  )
  target_include_directories(${target} PRIVATE ${WINML_TEST_INC_DIR})
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

  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  source_group(TREE ${WINML_TEST_SRC_DIR} FILES ${_UT_SOURCES})
  set_winml_target_properties(${_UT_TARGET})
  target_compile_definitions(${_UT_TARGET} PRIVATE BUILD_GOOGLE_TEST)

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif()
  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest winml_google_test_lib ${onnxruntime_EXTERNAL_LIBRARIES} winml_lib_common onnxruntime)

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
    file(GLOB winml_test_scenario_src CONFIGURE_DEPENDS "${winml_test_src_path}/scenario/cppwinrt/*.cpp")
    set(${output_winml_test_scenario_libs} "onnxruntime_providers_dml" PARENT_SCOPE)
  else()
    set(winml_test_scenario_src "${winml_test_src_path}/scenario/cppwinrt/scenariotestscppwinrt.cpp")
  endif()
  set(${output_winml_test_scenario_src} ${winml_test_scenario_src} PARENT_SCOPE)
endfunction()

function(get_winml_test_api_src
  winml_test_src_path
  output_winml_test_api_src
)
  file(GLOB winml_test_api_src CONFIGURE_DEPENDS "${winml_test_src_path}/api/*.cpp")
  set(${output_winml_test_api_src} ${winml_test_api_src} PARENT_SCOPE)
endfunction()

file(GLOB winml_test_common_src CONFIGURE_DEPENDS "${WINML_TEST_SRC_DIR}/common/*.cpp")
add_library(winml_test_common STATIC ${winml_test_common_src})
add_dependencies(winml_test_common
  onnx
  winml_api
  winml_dll
)

add_library(winml_google_test_lib STATIC ${WINML_TEST_SRC_DIR}/common/googletest/main.cpp)
set_winml_target_properties(winml_google_test_lib)

set_winml_target_properties(winml_test_common)
get_winml_test_api_src(${WINML_TEST_SRC_DIR} winml_test_api_src)
add_winml_test(
  TARGET winml_test_api
  SOURCES ${winml_test_api_src}
  LIBS winml_test_common delayimp.lib
)
target_precompiled_header(winml_test_api testPch.h)

target_link_options(winml_test_api PRIVATE /DELAYLOAD:dxgi.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:api-ms-win-core-file-l1-2-2.dll /DELAYLOAD:api-ms-win-core-synch-l1-2-1.dll)
if (onnxruntime_USE_DML)
  target_link_options(winml_test_api PRIVATE /DELAYLOAD:directml.dll)
endif()
if (EXISTS ${dxcore_header})
  target_link_options(winml_test_api PRIVATE /DELAYLOAD:ext-ms-win-dxcore-l1-*.dll)
endif()

get_winml_test_scenario_src(${WINML_TEST_SRC_DIR} winml_test_scenario_src winml_test_scenario_libs)
add_winml_test(
  TARGET winml_test_scenario
  SOURCES ${winml_test_scenario_src}
  LIBS winml_test_common delayimp.lib ${winml_test_scenario_libs}
)
target_precompiled_header(winml_test_scenario testPch.h)

target_link_options(winml_test_scenario PRIVATE /DELAYLOAD:d2d1.dll /DELAYLOAD:d3d11.dll /DELAYLOAD:dxgi.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:api-ms-win-core-libraryloader-l1-2-1.dll /DELAYLOAD:api-ms-win-core-file-l1-2-2.dll /DELAYLOAD:api-ms-win-core-synch-l1-2-1.dll)
if (onnxruntime_USE_DML)
  target_link_options(winml_test_scenario PRIVATE /DELAYLOAD:directml.dll)
endif()
if (EXISTS ${dxcore_header})
  target_link_options(winml_test_scenario PRIVATE /DELAYLOAD:ext-ms-win-dxcore-l1-*.dll)
endif()

# necessary for winml_test_scenario because of a still unknown reason, api-ms-win-core-libraryloader-l1-2-1.dll is linked against
# on dev machines but not on the aiinfra agent pool
target_link_options(winml_test_scenario PRIVATE /ignore:4199)


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
add_winml_collateral("${WINML_TEST_SRC_DIR}/collateral/images/*.png")
add_winml_collateral("${WINML_TEST_SRC_DIR}/collateral/models/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/common/testdata/squeezenet/*")
add_winml_collateral("${WINML_TEST_SRC_DIR}/scenario/cppwinrt/*.onnx")
add_winml_collateral("${WINML_TEST_SRC_DIR}/scenario/models/*.onnx")
add_winml_collateral("${REPO_ROOT}/onnxruntime/test/testdata/sequence_length.onnx")
add_winml_collateral("${REPO_ROOT}/onnxruntime/test/testdata/sequence_construct.onnx")
