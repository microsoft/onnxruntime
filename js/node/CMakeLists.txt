cmake_minimum_required(VERSION 3.11)

project (onnxruntime-node)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(NAPI_VERSION=${napi_build_version})

# dist variables
execute_process(COMMAND node -e "console.log(process.arch)"
                OUTPUT_VARIABLE node_arch OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND node -e "console.log(process.platform)"
                OUTPUT_VARIABLE node_platform OUTPUT_STRIP_TRAILING_WHITESPACE)
file(READ ${CMAKE_SOURCE_DIR}/../../VERSION_NUMBER ort_version)
string(STRIP "${ort_version}" ort_version)
set(dist_folder "${CMAKE_SOURCE_DIR}/bin/napi-v3/${node_platform}/${node_arch}/")

# onnxruntime.dll dir
if(NOT ONNXRUNTIME_BUILD_DIR)
  if (WIN32)
    set(ONNXRUNTIME_BUILD_DIR ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE})
  else()
    set(ONNXRUNTIME_BUILD_DIR ${CMAKE_SOURCE_DIR}/../../build/Linux)
  endif()
endif()

# include dirs
include_directories(${CMAKE_JS_INC})
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime/core/session)
include_directories(${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

# source files
file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

add_library(onnxruntime_binding SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(onnxruntime_binding PROPERTIES
  PREFIX "" SUFFIX ".node"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH FALSE)
target_link_libraries(onnxruntime_binding PRIVATE ${CMAKE_JS_LIB})

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
