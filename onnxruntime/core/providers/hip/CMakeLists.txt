# Minimum CMake required
cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_providers_hip CXX)

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

#set(onnxruntime_EXTERNAL_DEPENDENCIES onnx_proto re2)

option(onnxruntime_USE_HIP "Build with AMD GPU support" OFF)
set(onnxruntime_HIP_HOME /opt/rocm)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../../..)

if (onnxruntime_USE_HIP)
  if (WIN32)
    message(FATAL_ERROR "HIP does not support build in Windows!")
  endif()
  set(CXX_EXTENSIONS OFF)
  set(AMD_HIP_HOME ${onnxruntime_HIP_HOME})
  add_definitions(-DUSE_HIP=1)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__=1")

  set(ONNXRUNTIME_HIP_LIBRARIES ${HIP_LIBRARIES})
  list(APPEND ONNXRUNTIME_HIP_LIBRARIES hipblas)
endif()

if (onnxruntime_USE_HIP)
  # Add search paths for default hip installation
  list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_HIP_HOME} ${onnxruntime_HIP_HOME}/hip ${onnxruntime_HIP_HOME}/hcc ${onnxruntime_HIP_HOME}/hipblas)

  set(CMAKE_MODULE_PATH "${onnxruntime_HIP_HOME}/hip/cmake" ${CMAKE_MODULE_PATH})
  #find_package(HIP)
  find_library(HIP_LIB hip_hcc)
  find_library(HIP_BLAS hipblas)
  set(HIP_LIBS ${HIP_LIB} ${HIP_BLAS})

  file(GLOB_RECURSE onnxruntime_providers_hip_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/hip/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/hip/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_hip_cc_srcs})
  add_library(onnxruntime_providers_hip ${onnxruntime_providers_hip_cc_srcs})
  target_link_libraries(onnxruntime_providers_hip PRIVATE  ${HIP_LIBS})
  set_target_properties(onnxruntime_providers_hip PROPERTIES FOLDER "ONNXRuntime")
  target_compile_options(onnxruntime_providers_hip PRIVATE -Wno-error=sign-compare)
  target_include_directories(onnxruntime_providers_hip PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_HIP_HOME}/include)
  #onnxruntime_add_include_to_target(onnxruntime_providers_hip onnxruntime_common onnxruntime_framework onnx)
  #add_dependencies(onnxruntime_providers_hip ${onnxruntime_EXTERNAL_DEPENDENCIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/hip  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_hip PROPERTIES LINKER_LANGUAGE CXX)
endif()