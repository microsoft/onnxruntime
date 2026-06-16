# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_MIGRAPHX=1)
  include_directories(${protobuf_SOURCE_DIR} ${eigen_SOURCE_DIR} ${onnx_SOURCE_DIR})
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()

  # Add search paths for default rocm installation
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm $ENV{HIP_PATH})

  if(POLICY CMP0144)
      # Suppress the warning about the small capitals of the package name
      cmake_policy(SET CMP0144 NEW)
  endif()

  if(WIN32 AND NOT HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
  endif()

  find_package(hip REQUIRED)
  find_package(migraphx REQUIRED PATHS ${AMD_MIGRAPHX_HOME})

  file(GLOB_RECURSE onnxruntime_providers_migraphx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_migraphx ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_common onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  add_dependencies(onnxruntime_providers_migraphx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE migraphx::c hip::host ${ONNXRUNTIME_PROVIDERS_SHARED} onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  target_include_directories(onnxruntime_providers_migraphx PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/migraphx/onnxruntime)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_migraphx PRIVATE ONNXIFI_BUILD_LIBRARY=1 ONNX_ML=1 ONNX_NAMESPACE=onnx)
  if(MSVC)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS /DEF:${ONNXRUNTIME_ROOT}/core/providers/migraphx/symbols.def)
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE ws2_32)
  else()
    target_compile_options(onnxruntime_providers_migraphx PRIVATE -Wno-error=sign-compare)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
  endif()
  if(UNIX)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/migraphx/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE  stdc++fs)
  endif()

  set(CMAKE_REQUIRED_LIBRARIES migraphx::c)

  check_symbol_exists(migraphx_onnx_options_set_external_data_path
    "migraphx/migraphx.h" HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH)

  if(HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH)
    target_compile_definitions(onnxruntime_providers_migraphx PRIVATE HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH=1)
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_training)
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE onnxruntime_training)
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_migraphx Python::Module)
    endif()
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    foreach(file migraphx-hiprtc-driver.exe migraphx.dll migraphx_c.dll migraphx_cpu.dll migraphx_device.dll migraphx_gpu.dll migraphx_onnx.dll migraphx_tf.dll)
      set(_source "${AMD_MIGRAPHX_HOME}/bin/${file}")
      if(EXISTS "${_source}")
        add_custom_command(TARGET onnxruntime_providers_migraphx
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_source} $<TARGET_FILE_DIR:onnxruntime_providers_migraphx>)
        set(_target "$<TARGET_FILE_DIR:onnxruntime_providers_migraphx>/${file}")
        list(APPEND _migraphx_targets ${_target})
      endif()
    endforeach()
    set(MIGRAPHX_LIB_FILES ${_migraphx_targets} CACHE INTERNAL "" FORCE)
    install(FILES ${_migraphx_targets}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
    get_property(_amdhip64_location TARGET hip::amdhip64 PROPERTY IMPORTED_LOCATION_RELEASE)
    cmake_path(GET _amdhip64_location PARENT_PATH _hipsdk_path)
    foreach(file amd_comgr0602.dll amd_comgr0604.dll amd_comgr0700.dll hiprtc0602.dll hiprtc0604.dll hiprtc0700.dll hiprtc-builtins0602.dll hiprtc-builtins0604.dll hiprtc-builtins0700.dll)
      set(_source "${_hipsdk_path}/${file}")
      if(EXISTS "${_source}")
        add_custom_command(TARGET onnxruntime_providers_migraphx
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_source} $<TARGET_FILE_DIR:onnxruntime_providers_migraphx>)
        set(_target "$<TARGET_FILE_DIR:onnxruntime_providers_migraphx>/${file}")
        list(APPEND _hipsdk_targets ${_target})
      endif()
    endforeach()
    set(HIPSDK_LIB_FILES ${_hipsdk_targets} CACHE INTERNAL "" FORCE)
    install(FILES ${_hipsdk_targets}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  install(TARGETS onnxruntime_providers_migraphx
          EXPORT onnxruntime_providers_migraphxTargets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
