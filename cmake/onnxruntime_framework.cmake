# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_framework_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.cc"
)

if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
file(GLOB_RECURSE onnxruntime_training_framework_torch_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.cc"
)

  list(APPEND onnxruntime_framework_srcs ${onnxruntime_training_framework_torch_srcs})
endif()

if (onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_framework_src_exclude
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
if(onnxruntime_USE_TENSORRT OR onnxruntime_USE_NCCL)
# TODO: for now, core framework depends on CUDA. It should be moved to TensorRT EP
# TODO: provider_bridge_ort.cc should not include nccl.h
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${onnxruntime_CUDNN_HOME}/include PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
endif()
# Needed for the provider interface, as it includes training headers when training is enabled
if (onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_framework PRIVATE ${ORTTRAINING_ROOT})
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    onnxruntime_add_include_to_target(onnxruntime_framework Python::Module)
    target_include_directories(onnxruntime_framework PRIVATE ${dlpack_SOURCE_DIR}/include)
  endif()
  if (onnxruntime_USE_NCCL OR onnxruntime_USE_MPI)
    target_include_directories(onnxruntime_framework PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  endif()
endif()

if (onnxruntime_ENABLE_ATEN)
  # DLPack is a header-only dependency
  set(DLPACK_INCLUDE_DIR ${dlpack_SOURCE_DIR}/include)
  target_include_directories(onnxruntime_framework PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
onnxruntime_add_include_to_target(onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers safeint_interface Boost::mp11)

if (onnxruntime_USE_MIMALLOC)
    target_link_libraries(onnxruntime_framework mimalloc-static)
endif()

if (onnxruntime_BUILD_WEBASSEMBLY)
  target_link_libraries(onnxruntime_framework ${ABSEIL_LIBS})
endif()

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

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB)
  find_package (SQLite3)
  if (SQLITE3_FOUND)
    include_directories(${SQLite3_INCLUDE_DIR})
    target_link_libraries (onnxruntime_framework ${SQLite3_LIBRARY})
  else()
    message( FATAL_ERROR "Could not locate SQLite3 package." )
  endif (SQLITE3_FOUND)
  target_compile_definitions(onnxruntime_framework PRIVATE DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB)
endif()

if (WIN32)
  target_compile_definitions(onnxruntime_framework PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_framework
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/framework  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
