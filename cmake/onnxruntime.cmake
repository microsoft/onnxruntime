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

# Gets the public C/C++ API header files
function(get_c_cxx_api_headers HEADERS_VAR)
  set(_headers
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_float16.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"
  )

  if (onnxruntime_ENABLE_TRAINING_APIS)
    list(APPEND _headers "${REPO_ROOT}/orttraining/orttraining/training_api/include/onnxruntime_training_c_api.h")
    list(APPEND _headers "${REPO_ROOT}/orttraining/orttraining/training_api/include/onnxruntime_training_cxx_api.h")
    list(APPEND _headers "${REPO_ROOT}/orttraining/orttraining/training_api/include/onnxruntime_training_cxx_inline.h")
  endif()

  # need to add header files for enabled EPs
  foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
    file(GLOB _provider_headers CONFIGURE_DEPENDS
      "${REPO_ROOT}/include/onnxruntime/core/providers/${f}/*.h"
    )
    list(APPEND _headers ${_provider_headers})
  endforeach()

  set(${HEADERS_VAR} ${_headers} PARENT_SCOPE)
endfunction()

get_c_cxx_api_headers(ONNXRUNTIME_PUBLIC_HEADERS)

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
  # apple framework requires the header file be part of the library
  onnxruntime_add_shared_library(onnxruntime
    ${ONNXRUNTIME_PUBLIC_HEADERS}
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
    MACOSX_FRAMEWORK_INFO_PLIST ${INFO_PLIST_PATH}
    SOVERSION ${ORT_VERSION}
    # Note: The PUBLIC_HEADER and VERSION properties for the 'onnxruntime' target will be set later in this file.
  )
else()
  onnxruntime_add_shared_library(onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c)
  if (onnxruntime_USE_CUDA)
    set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\\$ORIGIN")
  endif()
endif()

add_dependencies(onnxruntime onnxruntime_generate_def ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime>")

target_compile_definitions(onnxruntime PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_MINOR=${VERSION_MINOR_PART})
target_compile_definitions(onnxruntime PRIVATE VER_BUILD=${VERSION_BUILD_PART})
target_compile_definitions(onnxruntime PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
target_compile_definitions(onnxruntime PRIVATE VER_STRING=\"${VERSION_STRING}\")
target_compile_definitions(onnxruntime PRIVATE FILE_NAME=\"onnxruntime.dll\")

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
  elseif (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
  endif()
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND onnxruntime_MINIMAL_BUILD)
  # target onnxruntime is a shared library, the dummy __cxa_demangle is only attach to it to avoid
  # affecting downstream ort library users with the behaviour of dummy __cxa_demangle. So the dummy
  # __cxa_demangle must not expose to libonnxruntime_common.a. It works as when the linker is
  # creating the DSO, our dummy __cxa_demangle always comes before libc++abi.a so the
  # __cxa_demangle in libc++abi.a is discarded, thus, huge binary size reduction.
  target_sources(onnxruntime PRIVATE "${ONNXRUNTIME_ROOT}/core/platform/android/cxa_demangle.cc")
  target_compile_definitions(onnxruntime PRIVATE USE_DUMMY_EXA_DEMANGLE=1)
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
  set(ANDROID_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/android/headers)
  file(MAKE_DIRECTORY ${ANDROID_HEADERS_DIR})
  # copy the header files one by one
  foreach(h_ ${ONNXRUNTIME_PUBLIC_HEADERS})
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
  ${PROVIDERS_QNN}
  ${PROVIDERS_SNPE}
  ${PROVIDERS_TVM}
  ${PROVIDERS_RKNPU}
  ${PROVIDERS_XNNPACK}
  ${PROVIDERS_WEBNN}
  ${PROVIDERS_AZURE}
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

if (onnxruntime_USE_EXTENSIONS)
  list(APPEND onnxruntime_INTERNAL_LIBRARIES
    onnxruntime_extensions
    ocos_operators
    noexcep_operators
  )
endif()

# If you are linking a new library, please add it to the list onnxruntime_INTERNAL_LIBRARIES or onnxruntime_EXTERNAL_LIBRARIES,
# Please do not add a library directly to the target_link_libraries command
target_link_libraries(onnxruntime PRIVATE
    ${onnxruntime_INTERNAL_LIBRARIES}
    ${onnxruntime_EXTERNAL_LIBRARIES}
)

set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_DELAYLOAD_FLAGS})
set_target_properties(onnxruntime PROPERTIES
  PUBLIC_HEADER "${ONNXRUNTIME_PUBLIC_HEADERS}"
  LINK_DEPENDS ${SYMBOL_FILE}
  VERSION ${ORT_VERSION}
  FOLDER "ONNXRuntime"
)

install(TARGETS onnxruntime
        EXPORT ${PROJECT_NAME}Targets
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime
        ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})


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
  set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-${CMAKE_OSX_SYSROOT})

  # Setup the various directories required. Remove any existing ones so we start with a clean directory.
  set(STATIC_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/static_libraries)
  set(STATIC_LIB_TEMP_DIR ${STATIC_LIB_DIR}/temp)
  add_custom_command(TARGET onnxruntime PRE_BUILD COMMAND ${CMAKE_COMMAND} -E rm -rf ${STATIC_LIB_DIR})
  add_custom_command(TARGET onnxruntime PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_LIB_DIR})
  add_custom_command(TARGET onnxruntime PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_LIB_TEMP_DIR})

  set(STATIC_FRAMEWORK_DIR ${STATIC_FRAMEWORK_OUTPUT_DIR}/static_framework/onnxruntime.framework)
  add_custom_command(TARGET onnxruntime PRE_BUILD COMMAND ${CMAKE_COMMAND} -E rm -rf ${STATIC_FRAMEWORK_DIR})
  add_custom_command(TARGET onnxruntime PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_FRAMEWORK_DIR})

  # replicate XCode's Single Object Pre-Link
  # link the internal onnxruntime .o files with the external .a files into a single relocatable object
  # to enforce symbol visibility. doing it this way limits the symbols included from the .a files to symbols used
  # by the ORT .o files.

  # If it's an onnxruntime library, extract .o files to a separate directory for each library to avoid any clashes
  # with filenames (e.g. utils.o)
  foreach(_LIB ${onnxruntime_INTERNAL_LIBRARIES} )
    GET_TARGET_PROPERTY(_LIB_TYPE ${_LIB} TYPE)
    if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
      set(CUR_STATIC_LIB_OBJ_DIR ${STATIC_LIB_TEMP_DIR}/$<TARGET_LINKER_FILE_BASE_NAME:${_LIB}>)
      add_custom_command(TARGET onnxruntime POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E make_directory ${CUR_STATIC_LIB_OBJ_DIR})

      add_custom_command(TARGET onnxruntime POST_BUILD
                         COMMAND ar ARGS -x $<TARGET_FILE:${_LIB}>
                         WORKING_DIRECTORY ${CUR_STATIC_LIB_OBJ_DIR})
    endif()
  endforeach()

  # for external libraries we create a symlink to the .a file
  foreach(_LIB ${onnxruntime_EXTERNAL_LIBRARIES})
    GET_TARGET_PROPERTY(_LIB_TYPE ${_LIB} TYPE)
    if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
      add_custom_command(TARGET onnxruntime POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E create_symlink
                           $<TARGET_FILE:${_LIB}> ${STATIC_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:${_LIB}>)
    endif()
  endforeach()

  # do the pre-link with `ld -r` to create a single relocatable object with correct symbol visibility
  add_custom_command(TARGET onnxruntime POST_BUILD
                     COMMAND ld ARGS -r -o ${STATIC_LIB_DIR}/prelinked_objects.o */*.o ../*.a
                     WORKING_DIRECTORY ${STATIC_LIB_TEMP_DIR})

  # create the static library
  add_custom_command(TARGET onnxruntime POST_BUILD
                     COMMAND libtool -static -o ${STATIC_FRAMEWORK_DIR}/onnxruntime prelinked_objects.o
                     WORKING_DIRECTORY ${STATIC_LIB_DIR})

  # Assemble the other pieces of the static framework
  add_custom_command(TARGET onnxruntime POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                       copy_if_different ${INFO_PLIST_PATH} ${STATIC_FRAMEWORK_DIR}/Info.plist)

  # add the framework header files
  set(STATIC_FRAMEWORK_HEADER_DIR ${STATIC_FRAMEWORK_DIR}/Headers)
  file(MAKE_DIRECTORY ${STATIC_FRAMEWORK_HEADER_DIR})

  foreach(h_ ${ONNXRUNTIME_PUBLIC_HEADERS})
    get_filename_component(HEADER_NAME_ ${h_} NAME)
    add_custom_command(TARGET onnxruntime POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E
                         copy_if_different ${h_} ${STATIC_FRAMEWORK_HEADER_DIR}/${HEADER_NAME_})
  endforeach()

endif()
