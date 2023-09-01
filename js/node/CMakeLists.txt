cmake_minimum_required(VERSION 3.11)

project (onnxruntime-node)

set(CMAKE_CXX_STANDARD 14)

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
option(USE_CUDA "Build with CUDA support" OFF)
option(USE_TENSORRT "Build with TensorRT support" OFF)
option(USE_COREML "Build with CoreML support" OFF)

if(USE_DML)
  add_compile_definitions(USE_DML=1)
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

# source files
file(GLOB ORT_NODEJS_BINDING_SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

add_library(onnxruntime_binding SHARED ${ORT_NODEJS_BINDING_SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(onnxruntime_binding PROPERTIES
  PREFIX "" SUFFIX ".node"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH FALSE)
target_link_libraries(onnxruntime_binding PRIVATE ${CMAKE_JS_LIB})

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# add libraries
if (WIN32)
  target_link_directories(onnxruntime_binding PRIVATE ${ONNXRUNTIME_BUILD_DIR}/${CMAKE_BUILD_TYPE})
else()
  target_link_directories(onnxruntime_binding PRIVATE ${ONNXRUNTIME_BUILD_DIR})
endif()

if (WIN32)
  target_link_libraries(onnxruntime_binding PRIVATE onnxruntime.lib)
elseif (APPLE)
  target_link_libraries(onnxruntime_binding PRIVATE libonnxruntime.${ort_version}.dylib)
  set_target_properties(onnxruntime_binding PROPERTIES INSTALL_RPATH "@loader_path")
else()
  target_link_libraries(onnxruntime_binding PRIVATE libonnxruntime.so.${ort_version})
  set_target_properties(onnxruntime_binding PROPERTIES INSTALL_RPATH "$ORIGIN/")
endif()

# post build

add_custom_command(
  TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onnxruntime_binding> ${dist_folder}
)

if (WIN32)
  add_custom_command(
    TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_BUILD_DIR}/${CMAKE_BUILD_TYPE}/onnxruntime.dll
      ${dist_folder}
  )
  if (USE_DML)
    add_custom_command(
      TARGET onnxruntime_binding POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_BUILD_DIR}/${CMAKE_BUILD_TYPE}/DirectML.dll
      ${dist_folder}
    )
  endif ()
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(
      TARGET onnxruntime_binding POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        ${ONNXRUNTIME_BUILD_DIR}/${CMAKE_BUILD_TYPE}/onnxruntime.pdb
        ${dist_folder}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE_DIR:onnxruntime_binding>/onnxruntime_binding.pdb ${dist_folder}
    )
  endif()
elseif (APPLE)
  add_custom_command(
    TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime.${ort_version}.dylib
      ${dist_folder}
  )
elseif (UNIX)
  add_custom_command(
    TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime.so.${ort_version}
      ${dist_folder}
  )
else()
  message(FATAL_ERROR "Platform not supported.")
endif()
