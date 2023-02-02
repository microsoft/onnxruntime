# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

message(STATUS "[onnxruntime-extensions] Building onnxruntime-extensions: ${onnxruntime_EXTENSIONS_PATH}")

# add compile definition to enable custom operators in onnxruntime-extensions
add_compile_definitions(ENABLE_EXTENSION_CUSTOM_OPS)

# set options for onnxruntime-extensions
set(OCOS_ENABLE_CTEST OFF CACHE INTERNAL "")
set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "")

# backup CMAKE_CXX_FLAGS in case we'll rewrite it for exceptions flag
set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")

# disable exceptions
if(onnxruntime_DISABLE_EXCEPTIONS)
  # ort-ext needs exceptions enabled for some of the 3rd party libraries.
  # ort-ext will provide a try/catch layer around all entry points and convert any exceptions to status messages
  # so no exceptions are passed up to ort
  set(OCOS_ENABLE_CPP_EXCEPTIONS ON CACHE INTERNAL "")
  string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "-fno-unwind-tables" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "-fno-asynchronous-unwind-tables" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

# customize operators used
if (onnxruntime_REDUCED_OPS_BUILD)
  set(OCOS_ENABLE_SELECTED_OPLIST ON CACHE INTERNAL "")
endif()

if (onnxruntime_WEBASSEMBLY_DEFAULT_EXTENSION_FLAGS)
  set(OCOS_ENABLE_SPM_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_WORDPIECE_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_BERT_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_TF_STRING ON CACHE INTERNAL "")
  set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
  set(OCOS_ENABLE_BLINGFIRE OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_CV2 OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_OPENCV_CODECS OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_VISION OFF CACHE INTERNAL "")
endif()

# when onnxruntime-extensions is not a subdirectory of onnxruntime,
# output binary directory must be explicitly specified.
# and the output binary path is the same as CMake FetchContent pattern
add_subdirectory(${onnxruntime_EXTENSIONS_PATH} ${CMAKE_BINARY_DIR}/_deps/extensions-subbuild EXCLUDE_FROM_ALL)

# target library or executable are defined in CMakeLists.txt of onnxruntime-extensions
target_include_directories(ocos_operators PRIVATE ${RE2_INCLUDE_DIR} ${json_SOURCE_DIR}/include)
target_include_directories(ortcustomops PUBLIC ${onnxruntime_EXTENSIONS_PATH}/includes)

if(onnxruntime_DISABLE_EXCEPTIONS)
  # if we rewrited CMAKE_CXX_FLAGS, we need to restore it
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")
endif()
