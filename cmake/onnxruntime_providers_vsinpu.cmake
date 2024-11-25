  add_definitions(-DUSE_VSINPU=1)
  file(GLOB_RECURSE onnxruntime_providers_vsinpu_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/vsinpu/builders/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vsinpu/builders/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/vsinpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vsinpu/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vsinpu_srcs})
  add_library(onnxruntime_providers_vsinpu ${onnxruntime_providers_vsinpu_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_vsinpu
    onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers Boost::mp11
    safeint_interface )
  add_dependencies(onnxruntime_providers_vsinpu ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_vsinpu PROPERTIES FOLDER "ONNXRuntime" LINKER_LANGUAGE CXX)
  target_include_directories(onnxruntime_providers_vsinpu PRIVATE ${ONNXRUNTIME_ROOT} $ENV{TIM_VX_INSTALL}/include)

  find_library(TIMVX_LIBRARY NAMES tim-vx PATHS $ENV{TIM_VX_INSTALL}/lib NO_DEFAULT_PATH)
  if(NOT TIMVX_LIBRARY)
    message(FATAL_ERROR "TIM-VX library is not found!")
  endif()

  if(CMAKE_CROSSCOMPILING)
    message(STATUS "VSINPU ep will be cross compiled.")
    if(EXISTS "$ENV{VIVANTE_SDK_DIR}/drivers")
      set(DRIVER_DIR "$ENV{VIVANTE_SDK_DIR}/drivers")
    elseif(EXISTS "$ENV{VIVANTE_SDK_DIR}/lib")
      set(DRIVER_DIR "$ENV{VIVANTE_SDK_DIR}/lib")
    else()
      message(FATAL_ERROR "Neither drivers nor lib directory exists in this VIVANTE_SDK_DIR.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wl,-rpath-link ${DRIVER_DIR} ${TIMVX_LIBRARY}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    target_link_libraries(onnxruntime_providers_vsinpu PRIVATE ${TIMVX_LIBRARY})
  endif()
