# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
  message(FATAL_ERROR "CoreML EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
endif()

add_compile_definitions(USE_COREML=1)

# Check if we can build the coremltools code for creating an mlpackage with an mlprogram.
# The coremltools source requires std::filesystem::path which is only available from iOS 13 on.
set(_enable_ML_PROGRAM ON)
if (IOS AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 13.0)
  message(WARNING "CoreML ML Program is not supported on iOS < 13.0. Excluding ML Program support from build.")
  set(_enable_ML_PROGRAM OFF)
elseif(LINUX)
  # uuid-dev is required. we don't bother installing on CIs as it's really for manual developer testing.
  find_library(LibUUID_LIBRARY NAMES uuid)
  find_path(LibUUID_INCLUDE_DIR NAMES uuid/uuid.h)
  if (NOT LibUUID_INCLUDE_DIR)
    message(STATUS "uuid/uuid.h was not found as is required for ML Program support. "
                    "Run `sudo apt install uuid-dev` if you need to test ML Program related CoreML EP code. ")
    set(_enable_ML_PROGRAM OFF)
  endif()
endif()

if (_enable_ML_PROGRAM)
  add_compile_definitions(COREML_ENABLE_MLPROGRAM=1)
endif()

# Compile CoreML proto definition to ${CMAKE_CURRENT_BINARY_DIR}/coreml_proto
set(COREML_PROTO_ROOT ${coremltools_SOURCE_DIR}/mlmodel/format)
file(GLOB coreml_proto_srcs "${COREML_PROTO_ROOT}/*.proto")

onnxruntime_add_static_library(coreml_proto ${coreml_proto_srcs})
target_include_directories(coreml_proto
                           PUBLIC $<TARGET_PROPERTY:${PROTOBUF_LIB},INTERFACE_INCLUDE_DIRECTORIES>
                           "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(coreml_proto
                           PUBLIC $<TARGET_PROPERTY:${PROTOBUF_LIB},INTERFACE_COMPILE_DEFINITIONS>)
set_target_properties(coreml_proto PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
set_target_properties(coreml_proto PROPERTIES COMPILE_FLAGS "-fvisibility-inlines-hidden")

set(_src_sub_dir "coreml_proto/")
onnxruntime_protobuf_generate(
  APPEND_PATH
  GEN_SRC_SUB_DIR ${_src_sub_dir}
  IMPORT_DIRS ${COREML_PROTO_ROOT}
  TARGET coreml_proto
)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS coreml_proto
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

# Add the .proto and generated .cc/.h files to the External/coreml_proto folder in Visual Studio.
# Separate source_group for each as the .proto files are in the repo and the .cc/.h files are generated in the build
# output directory.
set_target_properties(coreml_proto PROPERTIES FOLDER "External")
source_group(TREE ${COREML_PROTO_ROOT} PREFIX coreml_proto FILES ${coreml_proto_srcs})

# filter to the generated .cc/.h files
get_target_property(coreml_proto_generated_srcs coreml_proto SOURCES)
list(FILTER coreml_proto_generated_srcs INCLUDE REGEX "\.pb\.(h|cc)$")
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR} PREFIX coreml_proto_generated FILES ${coreml_proto_generated_srcs})

# These are shared utils,
# TODO, move this to a separated lib when used by EPs other than NNAPI and CoreML
file(GLOB_RECURSE onnxruntime_providers_shared_utils_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
)

file(GLOB onnxruntime_providers_coreml_public_headers CONFIGURE_DEPENDS
  "${ONNXRUNTIME_INCLUDE_DIR}/core/providers/coreml/*.h"
)

file(GLOB
  onnxruntime_providers_coreml_cc_srcs_top CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/coreml/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/coreml/*.cc"
)

# Add builder source code
file(GLOB_RECURSE
  onnxruntime_providers_coreml_cc_srcs_nested CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/*.cc"
)

if(_enable_ML_PROGRAM)
  # Add helpers to create mlpackage weights. limit to just the files we need to minimize the changes to make them
  # build on Windows and Linux.
  file(GLOB
    onnxruntime_providers_coreml_milblob_cc_srcs CONFIGURE_DEPENDS
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/*.hpp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/*.cpp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/Util/*.hpp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/Blob/BlobDataType.hpp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/Blob/StorageFormat.hpp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/Blob/FileWriter.?pp"
    "${coremltools_SOURCE_DIR}/mlmodel/src/MILBlob/Blob/StorageWriter.?pp"
  )

  # Add helpers to create mlpackage
  file(GLOB
    onnxruntime_providers_coreml_modelpackage_cc_srcs CONFIGURE_DEPENDS
    "${coremltools_SOURCE_DIR}/modelpackage/src/ModelPackage.?pp"
    "${coremltools_SOURCE_DIR}/modelpackage/src/utils/JsonMap.?pp"
  )

  set(coremltools_srcs
    ${onnxruntime_providers_coreml_milblob_cc_srcs}
    ${onnxruntime_providers_coreml_modelpackage_cc_srcs}
  )

  source_group(TREE ${coremltools_SOURCE_DIR} PREFIX coremltools FILES ${coremltools_srcs})
endif()

# Add CoreML objective c++ source code
if (APPLE)
  file(GLOB
    onnxruntime_providers_coreml_objcc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model.mm"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils.mm"
  )
else()
  # add the Model implementation that uses the protobuf types but excludes any actual CoreML dependencies
  # by using stub implementations on non-Apple platforms.
  file(GLOB
    onnxruntime_providers_coreml_objcc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils_stub.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model_stub.cc"
  )
endif()

set(onnxruntime_providers_coreml_cc_srcs
  ${onnxruntime_providers_coreml_cc_srcs_top}
  ${onnxruntime_providers_coreml_cc_srcs_nested}
  ${onnxruntime_providers_shared_utils_cc_srcs}
  ${onnxruntime_providers_coreml_objcc_srcs}
)

source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_providers_coreml_cc_srcs})
source_group(TREE ${ONNXRUNTIME_INCLUDE_DIR} FILES ${onnxruntime_providers_coreml_public_headers})

onnxruntime_add_static_library(onnxruntime_providers_coreml
  ${onnxruntime_providers_coreml_public_headers}
  ${onnxruntime_providers_coreml_cc_srcs}
  ${coremltools_srcs}
)

onnxruntime_add_include_to_target(onnxruntime_providers_coreml
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11
  safeint_interface
)

onnxruntime_add_include_to_target(onnxruntime_providers_coreml coreml_proto)
target_link_libraries(onnxruntime_providers_coreml PRIVATE coreml_proto)
add_dependencies(onnxruntime_providers_coreml coreml_proto)

if (APPLE)
  target_compile_definitions(onnxruntime_providers_coreml PRIVATE __APPLE__)
endif()

if (_enable_ML_PROGRAM)
  # Setup coremltools fp16 and json dependencies for creating an mlpackage.
  #
  # These are also used by external/xnnpack.cmake. fp16 depends on psimd
  FetchContent_Declare(psimd URL ${DEP_URL_psimd} URL_HASH SHA1=${DEP_SHA1_psimd})
  onnxruntime_fetchcontent_makeavailable(psimd)
  set(PSIMD_SOURCE_DIR ${psimd_SOURCE_DIR})
  FetchContent_Declare(fp16 URL ${DEP_URL_fp16} URL_HASH SHA1=${DEP_SHA1_fp16})
  set(FP16_BUILD_TESTS OFF CACHE INTERNAL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
  onnxruntime_fetchcontent_makeavailable(fp16)

  # need to tweak the include paths to match what the coreml source code expects
  target_include_directories(onnxruntime_providers_coreml PRIVATE
                            ${fp16_SOURCE_DIR}/include
                            ${nlohmann_json_SOURCE_DIR}/single_include/nlohmann
                            ${coremltools_SOURCE_DIR}
                            ${coremltools_SOURCE_DIR}/mlmodel/src/
                            ${coremltools_SOURCE_DIR}/modelpackage/src/
  )

  add_dependencies(onnxruntime_providers_coreml nlohmann_json::nlohmann_json fp16)

  if (LINUX)
    target_link_libraries(onnxruntime_providers_coreml PRIVATE uuid)
  endif()
endif()

if (APPLE)
  target_link_libraries(onnxruntime_providers_coreml PRIVATE "-framework Foundation" "-framework CoreML")
endif()

add_dependencies(onnxruntime_providers_coreml ${onnxruntime_EXTERNAL_DEPENDENCIES})

set_target_properties(onnxruntime_providers_coreml PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_coreml PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_coreml PRIVATE ${ONNXRUNTIME_ROOT} ${coreml_INCLUDE_DIRS})
set_target_properties(onnxruntime_providers_coreml PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_coreml
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
