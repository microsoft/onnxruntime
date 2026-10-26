cmake_minimum_required(VERSION 3.26)
if(NOT EP_OP_CONFORMANCE_PARENT_BUILD)
  project(ort_ep_op_conformance_pilot LANGUAGES CXX)
endif()

include(CTest)

add_executable(ep_op_conformance_comparison_test runner/comparison_test.cc)
set_target_properties(ep_op_conformance_comparison_test PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME ep_op_conformance_comparison COMMAND ep_op_conformance_comparison_test)

set(ORT_PUBLIC_INCLUDE_DIR "" CACHE PATH "Directory containing public onnxruntime_cxx_api.h")
set(JSON_INCLUDE_DIR "" CACHE PATH "Directory containing nlohmann/json.hpp")
set(ONNX_PROTO_INCLUDE_DIR "" CACHE PATH "Directory containing generated onnx/onnx-ml.pb.h")
set(PROTOBUF_INCLUDE_DIR "" CACHE PATH "Matching protobuf include directory")
set(ONNX_PROTO_LIBRARY "" CACHE FILEPATH "Prebuilt ONNX protobuf library")
set(PROTOBUF_LIBRARY "" CACHE FILEPATH "Matching protobuf library")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/artifact.json")
  set(ORT_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

option(EP_OP_CONFORMANCE_FETCH_DEPS "Build pinned public ONNX, protobuf and JSON dependencies" OFF)
if(EP_OP_CONFORMANCE_PARENT_BUILD)
  set(ORT_PUBLIC_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/../include/onnxruntime/core/session")
elseif(EP_OP_CONFORMANCE_FETCH_DEPS)
  include(FetchContent)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
  set(protobuf_INSTALL ON CACHE BOOL "" FORCE)
  FetchContent_Declare(protobuf
    URL https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/v21.12
    URL_HASH SHA256=22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(protobuf)

  set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
  set(ONNX_ML ON CACHE BOOL "" FORCE)
  set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "" FORCE)
  set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(onnx
    URL https://codeload.github.com/onnx/onnx/tar.gz/refs/tags/v1.17.0
    URL_HASH SHA256=8d5e983c36037003615e5a02d36b18fc286541bf52de1a78f6cf9f32005a820e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_Declare(nlohmann_json
    URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnx nlohmann_json)
  if(NOT ORT_PUBLIC_INCLUDE_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime_cxx_api.h")
      set(ORT_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
    else()
      get_filename_component(ORT_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/onnxruntime/core/session" ABSOLUTE)
    endif()
  endif()
else()
  foreach(required JSON_INCLUDE_DIR ONNX_PROTO_INCLUDE_DIR PROTOBUF_INCLUDE_DIR ONNX_PROTO_LIBRARY PROTOBUF_LIBRARY)
    if(NOT ${required})
      message(FATAL_ERROR "Set ${required} to a matching public SDK/dependency build")
    endif()
  endforeach()
endif()
if(NOT EXISTS "${ORT_PUBLIC_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ORT_PUBLIC_INCLUDE_DIR to the public ONNX Runtime headers")
endif()

add_executable(onnxruntime_ep_op_conformance_test runner/main.cc)
set_target_properties(onnxruntime_ep_op_conformance_test PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(onnxruntime_ep_op_conformance_test PRIVATE ORT_API_MANUAL_INIT)
target_include_directories(onnxruntime_ep_op_conformance_test SYSTEM PRIVATE "${ORT_PUBLIC_INCLUDE_DIR}")
if(EP_OP_CONFORMANCE_PARENT_BUILD)
  target_include_directories(onnxruntime_ep_op_conformance_test SYSTEM PRIVATE "${ONNX_INCLUDE_DIRS}")
  target_link_libraries(onnxruntime_ep_op_conformance_test PRIVATE onnx_proto ${PROTOBUF_LIB} nlohmann_json::nlohmann_json)
elseif(EP_OP_CONFORMANCE_FETCH_DEPS)
  target_compile_definitions(onnxruntime_ep_op_conformance_test PRIVATE ONNX_ML=1 ONNX_API=)
  target_include_directories(onnxruntime_ep_op_conformance_test SYSTEM PRIVATE "${onnx_BINARY_DIR}")
  target_link_libraries(onnxruntime_ep_op_conformance_test PRIVATE onnx_proto protobuf::libprotobuf-lite nlohmann_json::nlohmann_json)
else()
  target_include_directories(onnxruntime_ep_op_conformance_test SYSTEM PRIVATE
    "${JSON_INCLUDE_DIR}" "${ONNX_PROTO_INCLUDE_DIR}" "${PROTOBUF_INCLUDE_DIR}")
  target_link_libraries(onnxruntime_ep_op_conformance_test PRIVATE "${ONNX_PROTO_LIBRARY}" "${PROTOBUF_LIBRARY}")
endif()
target_link_libraries(onnxruntime_ep_op_conformance_test PRIVATE ${CMAKE_DL_LIBS})

if(MSVC)
  target_compile_options(ep_op_conformance_comparison_test PRIVATE /W4 /WX)
  target_compile_options(onnxruntime_ep_op_conformance_test PRIVATE /W4 /WX)
endif()

set(EP_OP_CONFORMANCE_TEST_RUNTIME "" CACHE FILEPATH "ORT runtime for optional producer/integration tests")
set(EP_OP_CONFORMANCE_TEST_PLUGIN "" CACHE FILEPATH "Matching example_plugin_ep library for integration tests")
set(EP_OP_CONFORMANCE_TEST_CPP_CASES "" CACHE PATH "Cases captured by the C++ exporter for integration tests")
if(EP_OP_CONFORMANCE_PARENT_BUILD)
  set(case_root "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(case_inputs "${CMAKE_CURRENT_SOURCE_DIR}/cases/pilot.json"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cases_from_json.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${case_inputs})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cases_from_json.py"
            --definitions "${CMAKE_CURRENT_SOURCE_DIR}/cases/pilot.json" --print-key
    OUTPUT_VARIABLE case_key OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  add_custom_command(
    OUTPUT "${case_root}/${case_key}.json" "${case_root}/current.json"
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cases_from_json.py"
          --definitions "${CMAKE_CURRENT_SOURCE_DIR}/cases/pilot.json" --build-cache "${case_root}"
    DEPENDS ${case_inputs}
    VERBATIM)
  add_custom_target(ep_op_conformance_cases
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cases_from_json.py"
          --definitions "${CMAKE_CURRENT_SOURCE_DIR}/cases/pilot.json" --build-cache "${case_root}" --verify
    DEPENDS "${case_root}/${case_key}.json" "${case_root}/current.json"
    VERBATIM)
  add_dependencies(onnxruntime_ep_op_conformance_test ep_op_conformance_cases onnxruntime)
endif()
if(BUILD_TESTING AND EP_OP_CONFORMANCE_TEST_RUNTIME)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  set(pilot_test_args --runner $<TARGET_FILE:onnxruntime_ep_op_conformance_test>
                      --runtime "${EP_OP_CONFORMANCE_TEST_RUNTIME}")
  if(EP_OP_CONFORMANCE_TEST_PLUGIN)
    list(APPEND pilot_test_args --plugin-library "${EP_OP_CONFORMANCE_TEST_PLUGIN}")
  endif()
  if(EP_OP_CONFORMANCE_TEST_CPP_CASES)
    list(APPEND pilot_test_args --exported-cases "${EP_OP_CONFORMANCE_TEST_CPP_CASES}")
  endif()
  add_test(NAME ep_op_conformance_pilot
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_runner.py" ${pilot_test_args})
endif()