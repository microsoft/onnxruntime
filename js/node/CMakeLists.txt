cmake_minimum_required(VERSION 3.11)

project (onnxruntime-node)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(NAPI_VERSION=${napi_build_version})
add_compile_definitions(ORT_API_MANUAL_INIT)

# dist variables
execute_process(COMMAND node -e "console.log(process.platform)"
                OUTPUT_VARIABLE node_platform OUTPUT_STRIP_TRAILING_WHITESPACE)
file(READ ${CMAKE_SOURCE_DIR}/../../VERSION_NUMBER ort_version)
string(STRIP "${ort_version}" ort_version)
set(dist_folder "${CMAKE_SOURCE_DIR}/bin/napi-v3/${node_platform}/${NODE_ARCH}/")

# onnxruntime.dll dir
if(NOT ONNXRUNTIME_BUILD_DIR)
  if (WIN32)
    set(ONNXRUNTIME_BUILD_DIR ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE})
  elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(ONNXRUNTIME_BUILD_DIR ${CMAKE_SOURCE_DIR}/../../build/MacOS/${CMAKE_BUILD_TYPE})
  else()
    set(ONNXRUNTIME_BUILD_DIR ${CMAKE_SOURCE_DIR}/../../build/Linux/${CMAKE_BUILD_TYPE})
  endif()
endif()

# include dirs
include_directories(${CMAKE_JS_INC})
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime/core/session)
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime)
include_directories(${CMAKE_SOURCE_DIR}/../../onnxruntime)
include_directories(${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

# optional providers
option(USE_DML "Build with DirectML support" OFF)
option(USE_WEBGPU "Build with WebGPU support" OFF)
option(USE_CUDA "Build with CUDA support" OFF)
option(USE_TENSORRT "Build with TensorRT support" OFF)
option(USE_COREML "Build with CoreML support" OFF)
option(USE_QNN "Build with QNN support" OFF)

if(USE_DML)
  add_compile_definitions(USE_DML=1)
endif()
if(USE_WEBGPU)
  add_compile_definitions(USE_WEBGPU=1)
endif()
if(USE_CUDA)
  add_compile_definitions(USE_CUDA=1)
endif()
if(USE_TENSORRT)
  add_compile_definitions(USE_TENSORRT=1)
endif()
if(USE_COREML)
  add_compile_definitions(USE_COREML=1)
endif()
if(USE_QNN)
  add_compile_definitions(USE_QNN=1)
endif()

# source files
file(GLOB ORT_NODEJS_BINDING_SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

add_library(onnxruntime_binding SHARED ${ORT_NODEJS_BINDING_SOURCE_FILES} ${CMAKE_JS_SRC})
file(MAKE_DIRECTORY ${dist_folder})

set_target_properties(onnxruntime_binding PROPERTIES
  PREFIX "" SUFFIX ".node"
  RUNTIME_OUTPUT_DIRECTORY ${dist_folder}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dist_folder}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dist_folder}
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dist_folder}
  LIBRARY_OUTPUT_DIRECTORY ${dist_folder}
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dist_folder}
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dist_folder}
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dist_folder}
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH FALSE)
target_link_libraries(onnxruntime_binding PRIVATE ${CMAKE_JS_LIB})

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if (WIN32)
  if (${ONNXRUNTIME_GENERATOR} MATCHES "Ninja")
    set(ONNXRUNTIME_WIN_BIN_DIR ${ONNXRUNTIME_BUILD_DIR})
  else()
    set(ONNXRUNTIME_WIN_BIN_DIR ${ONNXRUNTIME_BUILD_DIR}/${CMAKE_BUILD_TYPE})
  endif()
  message(STATUS "onnxruntime dist dir: ${ONNXRUNTIME_WIN_BIN_DIR}")
endif()
# add libraries
if (WIN32)
  target_link_directories(onnxruntime_binding PRIVATE ${ONNXRUNTIME_WIN_BIN_DIR})
else()
  target_link_directories(onnxruntime_binding PRIVATE ${ONNXRUNTIME_BUILD_DIR})
endif()

if (WIN32)
  target_link_libraries(onnxruntime_binding PRIVATE onnxruntime)
elseif (APPLE)
  target_link_libraries(onnxruntime_binding PRIVATE libonnxruntime.${ort_version}.dylib)
  set_target_properties(onnxruntime_binding PROPERTIES INSTALL_RPATH "@loader_path")
else()
  target_link_libraries(onnxruntime_binding PRIVATE onnxruntime)
  set_target_properties(onnxruntime_binding PROPERTIES INSTALL_RPATH "$ORIGIN/")
endif()


if (WIN32)
  file(COPY ${ONNXRUNTIME_WIN_BIN_DIR}/onnxruntime.dll
      DESTINATION ${dist_folder})
  if (USE_DML)
    file(COPY ${ONNXRUNTIME_WIN_BIN_DIR}/DirectML.dll
      DESTINATION ${dist_folder})
  endif ()
elseif (APPLE)
  file(COPY ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime.dylib
      DESTINATION ${dist_folder} FOLLOW_SYMLINK_CHAIN)
elseif (UNIX)
  file(COPY ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime.so
      DESTINATION ${dist_folder} FOLLOW_SYMLINK_CHAIN)
else()
  message(FATAL_ERROR "Platform not supported.")
endif()
