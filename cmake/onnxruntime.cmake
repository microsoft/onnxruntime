# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(UNIX)
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.lds)
  if(APPLE)
    set(OUTPUT_STYLE xcode)
  else()
    set(OUTPUT_STYLE gcc)
  endif()
else()
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_dll.def)
  set(OUTPUT_STYLE vc)
endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
  set(OUTPUT_STYLE xcode)
endif()

# This macro is to get the path of header files for mobile packaging, for iOS and Android
macro(get_mobile_api_headers _HEADERS)
  # include both c and cxx api
  set(${_HEADERS}
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
  )

  # need to add header files for enabled EPs
  foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
    file(GLOB _provider_headers CONFIGURE_DEPENDS
      "${REPO_ROOT}/include/onnxruntime/core/providers/${f}/*.h"
    )
    list(APPEND ${_HEADERS} "${_provider_headers}")
    unset(_provider_headers)
  endforeach()
endmacro()

#If you want to verify if there is any extra line in symbols.txt, run
# nm -C -g --defined libonnxruntime.so |grep -v '\sA\s' | cut -f 3 -d ' ' | sort
# after build

list(APPEND SYMBOL_FILES "${REPO_ROOT}/tools/ci_build/gen_def.py")
foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
  list(APPEND SYMBOL_FILES "${ONNXRUNTIME_ROOT}/core/providers/${f}/symbols.txt")
endforeach()

add_custom_command(OUTPUT ${SYMBOL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c
  COMMAND ${Python_EXECUTABLE} "${REPO_ROOT}/tools/ci_build/gen_def.py"
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
  get_mobile_api_headers(APPLE_FRAMEWORK_HEADERS)

  # apple framework requires the header file be part of the library
  onnxruntime_add_shared_library(onnxruntime
    ${APPLE_FRAMEWORK_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/generated_source.c"
  )

  # create Info.plist for the framework and podspec for CocoaPods (optional)
  set(MACOSX_FRAMEWORK_NAME "onnxruntime")
  set(MACOSX_FRAMEWORK_IDENTIFIER "com.microsoft.onnxruntime")
  # Need to include CoreML as a weaklink for CocoaPods package if the EP is enabled
  if(onnxruntime_USE_COREML)
    set(APPLE_WEAK_FRAMEWORK "\\\"CoreML\\\"")
  endif()
  set(INFO_PLIST_PATH "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")
  configure_file(${REPO_ROOT}/cmake/Info.plist.in ${INFO_PLIST_PATH})
  configure_file(
    ${REPO_ROOT}/tools/ci_build/github/apple/framework_info.json.template
    ${CMAKE_CURRENT_BINARY_DIR}/framework_info.json)
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
  if (onnxruntime_USE_CUDA)
    set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\\$ORIGIN")
  endif()
endif()

add_dependencies(onnxruntime onnxruntime_generate_def ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT})

target_compile_definitions(onnxruntime PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_MINOR=${VERSION_MINOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_BUILD=${VERSION_BUILD_PART})
target_compile_definitions(onnxruntime PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
target_compile_definitions(onnxruntime PRIVATE VER_STRING=\"${VERSION_STRING}\")

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
    set(ONNXRUNTIME_SO_LINK_FLAG " -Wl,-exported_symbols_list,${SYMBOL_FILE}")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
      set_target_properties(onnxruntime PROPERTIES
        SOVERSION ${ORT_VERSION}
        MACOSX_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        BUILD_WITH_INSTALL_NAME_DIR TRUE
        INSTALL_NAME_DIR @rpath)
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

# we need to copy C/C++ API headers to be packed into Android AAR package
if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND onnxruntime_BUILD_JAVA)
  get_mobile_api_headers(ANDROID_AAR_HEADERS)
  set(ANDROID_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/android/headers)
  file(MAKE_DIRECTORY ${ANDROID_HEADERS_DIR})
  # copy the header files one by one
  foreach(h_ ${ANDROID_AAR_HEADERS})
    get_filename_component(HEADER_NAME_ ${h_} NAME)
    add_custom_command(TARGET onnxruntime POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${h_} ${ANDROID_HEADERS_DIR}/${HEADER_NAME_})
  endforeach()
endif()

# This list is a reversed topological ordering of library dependencies.
# Earlier entries may depend on later ones. Later ones should not depend on earlier ones.
set(onnxruntime_INTERNAL_LIBRARIES
  onnxruntime_session
  ${onnxruntime_libs}
  ${PROVIDERS_ACL}
  ${PROVIDERS_ARMNN}
  ${PROVIDERS_COREML}
  ${PROVIDERS_DML}
  ${PROVIDERS_NNAPI}
  ${PROVIDERS_NUPHAR}
  ${PROVIDERS_TVM}
  ${PROVIDERS_RKNPU}
  ${PROVIDERS_ROCM}
  ${PROVIDERS_VITISAI}
  ${PROVIDERS_INTERNAL_TESTING}
  ${onnxruntime_winml}
  onnxruntime_optimizer
  onnxruntime_providers
  ${onnxruntime_tvm_libs}
  onnxruntime_framework
  onnxruntime_graph
  onnxruntime_util
  ${ONNXRUNTIME_MLAS_LIBS}
  onnxruntime_common
  onnxruntime_flatbuffers
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnxruntime_INTERNAL_LIBRARIES
    onnxruntime_language_interop
    onnxruntime_pyop
  )
endif()

# If you are linking a new library, please add it to the list onnxruntime_INTERNAL_LIBRARIES or onnxruntime_EXTERNAL_LIBRARIES,
# Please do not add a library directly to the target_link_libraries command
target_link_libraries(onnxruntime PRIVATE
    ${onnxruntime_INTERNAL_LIBRARIES}
    ${onnxruntime_EXTERNAL_LIBRARIES}
)

set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_DELAYLOAD_FLAGS})
set_target_properties(onnxruntime PROPERTIES LINK_DEPENDS ${SYMBOL_FILE})


set_target_properties(onnxruntime PROPERTIES VERSION ${ORT_VERSION})

install(TARGETS onnxruntime
        ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "ONNXRuntime")

if (WIN32 AND NOT CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
  # Workaround STL bug https://github.com/microsoft/STL/issues/434#issuecomment-921321254
  # Note that the workaround makes std::system_error crash before Windows 10

  # The linker warns "LNK4199: /DELAYLOAD:api-ms-win-core-heapl2-1-0.dll ignored; no imports found from api-ms-win-core-heapl2-1-0.dll"
  # when you're not using imports directly, even though the import exists in the STL and the DLL would have been linked without DELAYLOAD
  target_link_options(onnxruntime PRIVATE /DELAYLOAD:api-ms-win-core-heapl2-1-0.dll /ignore:4199)
endif()

if (winml_is_inbox)
  # Apply linking flags required by inbox static analysis tools
  target_link_options(onnxruntime PRIVATE ${os_component_link_flags_list})
  # Link *_x64/*_arm64 DLLs for the ARM64X forwarder
  function(duplicate_shared_library target new_target)
    get_target_property(sources ${target} SOURCES)
    get_target_property(compile_definitions ${target} COMPILE_DEFINITIONS)
    get_target_property(compile_options ${target} COMPILE_OPTIONS)
    get_target_property(include_directories ${target} INCLUDE_DIRECTORIES)
    get_target_property(link_libraries ${target} LINK_LIBRARIES)
    get_target_property(link_flags ${target} LINK_FLAGS)
    get_target_property(link_options ${target} LINK_OPTIONS)
    add_library(${new_target} SHARED ${sources})
    add_dependencies(${target} ${new_target})
    target_compile_definitions(${new_target} PRIVATE ${compile_definitions})
    target_compile_options(${new_target} PRIVATE ${compile_options})
    target_include_directories(${new_target} PRIVATE ${include_directories})
    target_link_libraries(${new_target} PRIVATE ${link_libraries})
    set_property(TARGET ${new_target} PROPERTY LINK_FLAGS "${link_flags}")
    target_link_options(${new_target} PRIVATE ${link_options})
  endfunction()
  if (WAI_ARCH STREQUAL x64 OR WAI_ARCH STREQUAL arm64)
    duplicate_shared_library(onnxruntime onnxruntime_${WAI_ARCH})
  endif()
endif()

# Assemble the Apple static framework (iOS and macOS)
if(onnxruntime_BUILD_APPLE_FRAMEWORK)
  set(STATIC_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/static_libraries)
  file(MAKE_DIRECTORY ${STATIC_LIB_DIR})

  # Remove the existing files in the STATIC_LIB_DIR folder
  file(GLOB _OLD_STATIC_LIBS ${STATIC_LIB_DIR}/*.a)
  file(REMOVE "${_OLD_STATIC_LIBS}")

  # Go through all the static libraries, and create symbolic links
  foreach(_LIB ${onnxruntime_INTERNAL_LIBRARIES} ${onnxruntime_EXTERNAL_LIBRARIES})
    GET_TARGET_PROPERTY(_LIB_TYPE ${_LIB} TYPE)
    if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
      add_custom_command(TARGET onnxruntime POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${_LIB}> ${STATIC_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:${_LIB}>)
    endif()
  endforeach()

  if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-${CMAKE_OSX_SYSROOT})
  else() # macOS
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # Assemble the static framework
  set(STATIC_FRAMEWORK_DIR ${STATIC_FRAMEWORK_OUTPUT_DIR}/static_framework/onnxruntime.framework)
  set(STATIC_FRAMEWORK_HEADER_DIR ${STATIC_FRAMEWORK_DIR}/Headers)
  file(MAKE_DIRECTORY ${STATIC_FRAMEWORK_DIR})
  # Remove all files under STATIC_FRAMEWORK_DIR (if any)
  file(GLOB_RECURSE _OLD_STATIC_FRAMEWORK ${STATIC_FRAMEWORK_DIR}/*.*)
  file(REMOVE "${_OLD_STATIC_FRAMEWORK}")

  file(MAKE_DIRECTORY ${STATIC_FRAMEWORK_HEADER_DIR})

  # copy the header files one by one, and the Info.plist
  foreach(h_ ${APPLE_FRAMEWORK_HEADERS})
    get_filename_component(HEADER_NAME_ ${h_} NAME)
    add_custom_command(TARGET onnxruntime POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${h_} ${STATIC_FRAMEWORK_HEADER_DIR}/${HEADER_NAME_})
  endforeach()
  add_custom_command(TARGET onnxruntime POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INFO_PLIST_PATH} ${STATIC_FRAMEWORK_DIR}/Info.plist)

  # link the static library
  add_custom_command(TARGET onnxruntime POST_BUILD COMMAND libtool -static -o ${STATIC_FRAMEWORK_DIR}/onnxruntime *.a WORKING_DIRECTORY ${STATIC_LIB_DIR})
endif()
