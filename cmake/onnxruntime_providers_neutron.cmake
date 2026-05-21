# Copyright (c) NXP. All rights reserved.

add_definitions(-DUSE_NEUTRON=1)
file(GLOB_RECURSE onnxruntime_providers_neutron_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/neutron/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/neutron/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_neutron_srcs})
onnxruntime_add_static_library(onnxruntime_providers_neutron ${onnxruntime_providers_neutron_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_neutron
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface Eigen3::Eigen
)

set(NEUTRON_DRIVER_LIB "NeutronDriver")

if("$ENV{OECORE_TARGET_ARCH}" MATCHES "aarch64")
  message(STATUS " Target arch: $ENV{OECORE_TARGET_ARCH}")
  add_definitions(-DNEUTRON_AARCH64=1)
  target_link_libraries(onnxruntime_providers_neutron PRIVATE ${NEUTRON_DRIVER_LIB})
endif()

add_dependencies(onnxruntime_providers_neutron ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_neutron PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_neutron PRIVATE
  ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}
)
install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/neutron/neutron_provider_factory.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)

set_target_properties(onnxruntime_providers_neutron PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_neutron
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
