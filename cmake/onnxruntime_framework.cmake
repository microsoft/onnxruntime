# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(onnxruntime_ENABLE_TRAINING AND NOT EXISTS "${PYTHON_LIBRARY_PATH}")
  set(PYTHON_LIBRARY_SUFFIX_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"from distutils import sysconfig as s; print(s.get_config_var('LDVERSION') or s.get_config_var('VERSION'))\""
    OUTPUT_VARIABLE PYTHON_LIBRARY_SUFFIX
    RETURN_VALUE PYTHON_LIBRARY_SUFFIX_NOT_FOUND)

  if(MSVC)
    # construct PYTHON_LIBRARY_PATH from PYTHON_INCLUDE_DIR.
    get_filename_component(_PYTHON_ROOT ${PYTHON_INCLUDE_DIR} DIRECTORY)
    set(PYTHON_LIBRARY_PATH
        "${_PYTHON_ROOT}/libs/Python${PYTHON_LIBRARY_SUFFIX}.lib")
  else()

    set(PYTHON_LIBDIR_NOT_FOUND false)
    exec_program("${PYTHON_EXECUTABLE}"
      ARGS "-c \"from distutils import sysconfig as s; print(s.get_config_var('LIBDIR') or '')\""
      OUTPUT_VARIABLE PYTHON_LIBDIR
      RETURN_VALUE PYTHON_LIBDIR_NOT_FOUND)
    message("PYTHON_LIBDIR: ${PYTHON_LIBDIR}")
    set(PYTHON_MULTIARCH_NOT_FOUND false)
    exec_program("${PYTHON_EXECUTABLE}"
      ARGS "-c \"from distutils import sysconfig as s; print(s.get_config_var('MULTIARCH') or '')\""
      OUTPUT_VARIABLE PYTHON_MULTIARCH
      RETURN_VALUE PYTHON_MULTIARCH_NOT_FOUND)
    message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}, PYTHON_MULTIARCH: ${PYTHON_MULTIARCH}")
    get_filename_component(_PYTHON_BIN_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
    if(PYTHON_MULTIARCH)
      set(_PYTHON_LIBS_SEARCH "${PYTHON_LIBDIR}/${PYTHON_MULTIARCH}" "${PYTHON_LIBDIR}" "${_PYTHON_BIN_DIR}/../lib")
    else()
      set(_PYTHON_LIBS_SEARCH "${PYTHON_LIBDIR}" "${_PYTHON_BIN_DIR}/../lib")
    endif()

    # searching for Python libs in ${_PYTHON_LIBS_SEARCH}")
    find_library(
      PYTHON_LIBRARY_PATH
      NAMES "libpython${PYTHON_LIBRARY_SUFFIX}.so"
      PATHS ${_PYTHON_LIBS_SEARCH}
      NO_DEFAULT_PATH)
    
      # If all else fails, just set the name/version and let the linker figure out the path.
      if(NOT PYTHON_LIBRARY_PATH)
        set(PYTHON_LIBRARY_PATH libpython${PYTHON_LIBRARY_SUFFIX}.so)
      endif()
  endif()

  message("PYTHON_LIBRARY_PATH ${PYTHON_LIBRARY_PATH}")
  # raise an error if the python libs are still not found.
  if(NOT EXISTS "${PYTHON_LIBRARY_PATH}")
      message(FATAL_ERROR "Python libraries not found")
  endif()
endif()

file(GLOB_RECURSE onnxruntime_framework_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.cc"
)

if (onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_framework_src_exclude
    "${ONNXRUNTIME_ROOT}/core/framework/provider_bridge_ort.cc"
    "${ONNXRUNTIME_ROOT}/core/framework/fallback_cpu_capability.h"
    "${ONNXRUNTIME_ROOT}/core/framework/fallback_cpu_capability.cc"
  )

  # custom ops support must be explicitly enabled in a minimal build. exclude if not.
  if (NOT onnxruntime_MINIMAL_BUILD_CUSTOM_OPS)
    list(APPEND onnxruntime_framework_src_exclude
      "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/customregistry.h"
      "${ONNXRUNTIME_ROOT}/core/framework/customregistry.cc"
    )
  endif()

  list(REMOVE_ITEM onnxruntime_framework_srcs ${onnxruntime_framework_src_exclude})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_framework_srcs})

onnxruntime_add_static_library(onnxruntime_framework ${onnxruntime_framework_srcs})
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_framework PRIVATE ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
if(onnxruntime_USE_TENSORRT)
# TODO: for now, core framework depends on CUDA. It should be moved to TensorRT EP
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${onnxruntime_CUDNN_HOME}/include PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
endif()
# Needed for the provider interface, as it includes training headers when training is enabled
if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_framework PRIVATE ${ORTTRAINING_ROOT} ${PYTHON_INCLUDE_DIR})
  target_link_libraries(onnxruntime_framework PRIVATE ${PYTHON_LIBRARY_PATH})
  if (onnxruntime_USE_NCCL OR onnxruntime_USE_MPI)  
    target_include_directories(onnxruntime_framework PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  endif()
endif()
onnxruntime_add_include_to_target(onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
set_target_properties(onnxruntime_framework PROPERTIES FOLDER "ONNXRuntime")
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_framework ${onnxruntime_EXTERNAL_DEPENDENCIES})

# In order to find the shared provider libraries we need to add the origin to the rpath for all executables we build
# For the shared onnxruntime library, this is set in onnxruntime.cmake through CMAKE_SHARED_LINKER_FLAGS
# But our test files don't use the shared library so this must be set for them.
# For Win32 it generates an absolute path for shared providers based on the location of the executable/onnxruntime.dll
if (UNIX AND NOT APPLE AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_BUILD_WEBASSEMBLY)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
endif()

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_framework PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()


install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/framework  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
