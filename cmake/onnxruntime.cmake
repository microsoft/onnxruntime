# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(UNIX)
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.lds)
  set(OUTPUT_STYLE gcc)
else()
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_dll.def)
  set(OUTPUT_STYLE vc)
endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
  set(OUTPUT_STYLE xcode)
endif()

#If you want to verify if there is any extra line in symbols.txt, run
# nm -C -g --defined libonnxruntime.so |grep -v '\sA\s' | cut -f 3 -d ' ' | sort
# after build

list(APPEND SYMBOL_FILES "${REPO_ROOT}/tools/ci_build/gen_def.py")
foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
  list(APPEND SYMBOL_FILES "${ONNXRUNTIME_ROOT}/core/providers/${f}/symbols.txt")
endforeach()

add_custom_command(OUTPUT ${SYMBOL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c
  COMMAND ${PYTHON_EXECUTABLE} "${REPO_ROOT}/tools/ci_build/gen_def.py"
    --version_file "${ONNXRUNTIME_ROOT}/../VERSION_NUMBER" --src_root "${ONNXRUNTIME_ROOT}"
    --config ${ONNXRUNTIME_PROVIDER_NAMES} --style=${OUTPUT_STYLE} --output ${SYMBOL_FILE}
    --output_source ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c
  DEPENDS ${SYMBOL_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(onnxruntime_generate_def ALL DEPENDS ${SYMBOL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c)
if(WIN32)
  onnxruntime_add_shared_library(onnxruntime
    ${SYMBOL_FILE}
    "${ONNXRUNTIME_ROOT}/core/dll/dllmain.cc"
    "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc"
  )
elseif(onnxruntime_BUILD_APPLE_FRAMEWORK)
  # include both c and cxx api
  set(APPLE_FRAMEWORK_HEADERS
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
  )

  # need to add header files for enabled EPs
  foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
    file(GLOB _provider_headers CONFIGURE_DEPENDS
      "${REPO_ROOT}/include/onnxruntime/core/providers/${f}/*.h"
    )
    list(APPEND APPLE_FRAMEWORK_HEADERS "${_provider_headers}")
    unset(_provider_headers)
  endforeach()

  # apple framework requires the header file be part of the library
  onnxruntime_add_shared_library(onnxruntime
    ${APPLE_FRAMEWORK_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/generated_source.c"
  )

  set(MACOSX_FRAMEWORK_NAME "onnxruntime")
  set(MACOSX_FRAMEWORK_IDENTIFIER "com.microsoft.onnxruntime")
  configure_file(${REPO_ROOT}/cmake/Info.plist.in ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)

  set_target_properties(onnxruntime PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    PUBLIC_HEADER "${APPLE_FRAMEWORK_HEADERS}"
    MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist
    VERSION ${ORT_VERSION}
    SOVERSION  ${ORT_VERSION}
  )
else()
  onnxruntime_add_shared_library(onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c)
endif()

add_dependencies(onnxruntime onnxruntime_generate_def ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT})

target_compile_definitions(onnxruntime PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_MINOR=${VERSION_MINOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_BUILD=${VERSION_BUILD_PART})
target_compile_definitions(onnxruntime PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
target_compile_definitions(onnxruntime PRIVATE VER_STRING=\"${VERSION_STRING}\")

if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(UNIX)
  if (APPLE)
    set(ONNXRUNTIME_SO_LINK_FLAG " -Xlinker -dead_strip")
  else()
    set(ONNXRUNTIME_SO_LINK_FLAG " -Xlinker --version-script=${SYMBOL_FILE} -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
  endif()
else()
  set(ONNXRUNTIME_SO_LINK_FLAG " -DEF:${SYMBOL_FILE}")
endif()

if (NOT WIN32)
  if (APPLE OR ${CMAKE_SYSTEM_NAME} MATCHES "^iOS")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
      set_target_properties(onnxruntime PROPERTIES
        SOVERSION ${ORT_VERSION}
        MACOSX_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        BUILD_WITH_INSTALL_NAME_DIR TRUE
        INSTALL_NAME_DIR @rpath)
      set(ONNXRUNTIME_SO_LINK_FLAG " -Wl,-exported_symbols_list,${SYMBOL_FILE}")
    else()
        set_target_properties(onnxruntime PROPERTIES INSTALL_RPATH "@loader_path")
    endif()
  elseif (NOT onnxruntime_BUILD_WEBASSEMBLY)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
  endif()
endif()


# strip binary on Android, or for a minimal build on Unix
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR (onnxruntime_MINIMAL_BUILD AND UNIX))
  if (onnxruntime_MINIMAL_BUILD AND ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
    # don't strip
  else()
    set_target_properties(onnxruntime PROPERTIES LINK_FLAGS_RELEASE -s)
    set_target_properties(onnxruntime PROPERTIES LINK_FLAGS_MINSIZEREL -s)
  endif()
endif()

target_link_libraries(onnxruntime PRIVATE
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_MIGRAPHX}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_INTERNAL_TESTING}
    ${onnxruntime_winml}
    ${PROVIDERS_ROCM}
    ${PROVIDERS_COREML}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
    ${onnxruntime_EXTERNAL_LIBRARIES})

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_DELAYLOAD_FLAGS})
set_target_properties(onnxruntime PROPERTIES LINK_DEPENDS ${SYMBOL_FILE})


set_target_properties(onnxruntime PROPERTIES VERSION ${ORT_VERSION})

install(TARGETS onnxruntime
        ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "ONNXRuntime")

if (WINDOWS_STORE)
  target_link_options(onnxruntime PRIVATE /DELAYLOAD:api-ms-win-core-libraryloader-l1-2-1.dll)
endif()
