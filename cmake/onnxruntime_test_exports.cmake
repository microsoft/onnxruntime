# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

function(onnxruntime_export_test_symbols host)
  cmake_parse_arguments(EXPORTS "" "OBJECT_TARGET" "HOST_LIBS;MODULE_LIBS" ${ARGN})
  find_package(Python COMPONENTS Interpreter REQUIRED)

  set(module_files $<TARGET_OBJECTS:${EXPORTS_OBJECT_TARGET}>)
  set(host_files)
  set(library_dependencies)
  foreach(side HOST MODULE)
    foreach(library IN LISTS EXPORTS_${side}_LIBS)
      if(TARGET ${library})
        get_target_property(library_type ${library} TYPE)
        if(library_type STREQUAL "STATIC_LIBRARY")
          if(side STREQUAL "HOST")
            list(APPEND host_files $<TARGET_FILE:${library}>)
          else()
            list(APPEND module_files $<TARGET_FILE:${library}>)
          endif()
          list(APPEND library_dependencies ${library})
        elseif(side STREQUAL "MODULE")
          if(library_type STREQUAL "OBJECT_LIBRARY")
            list(APPEND module_files $<TARGET_OBJECTS:${library}>)
            list(APPEND library_dependencies ${library})
          elseif(library_type STREQUAL "UNKNOWN_LIBRARY")
            list(APPEND module_files $<TARGET_FILE:${library}>)
            list(APPEND library_dependencies ${library})
          endif()
        endif()
      endif()
    endforeach()
  endforeach()
  if(NOT host_files)
    message(FATAL_ERROR "No static host libraries found for ${host}'s test exports")
  endif()
  list(REMOVE_DUPLICATES host_files)
  list(REMOVE_DUPLICATES module_files)
  list(REMOVE_DUPLICATES library_dependencies)

  set(export_dir "${CMAKE_CURRENT_BINARY_DIR}/${host}_exports/$<CONFIG>")
  file(GENERATE OUTPUT "${export_dir}/host.rsp" CONTENT "\"$<JOIN:${host_files},\"\n\">\"\n")
  file(GENERATE OUTPUT "${export_dir}/module.rsp" CONTENT "\"$<JOIN:${module_files},\"\n\">\"\n")
  set(export_script "${REPO_ROOT}/tools/ci_build/gen_test_exports.py")
  set(export_file "${export_dir}/exports.def")
  set(force_include_file "${export_dir}/force_include.cc")
  add_custom_command(OUTPUT "${export_file}" "${force_include_file}"
    COMMAND ${Python_EXECUTABLE} "${export_script}"
      --linker "${CMAKE_LINKER}"
      --host "${export_dir}/host.rsp"
      --module "${export_dir}/module.rsp"
      --output "${export_file}"
      --force-include "${force_include_file}"
    DEPENDS "${export_script}" "${export_dir}/host.rsp" "${export_dir}/module.rsp"
      ${EXPORTS_OBJECT_TARGET} ${library_dependencies} ${host_files} ${module_files}
    VERBATIM)
  set_target_properties(${host} PROPERTIES ENABLE_EXPORTS ON WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  set_source_files_properties("${force_include_file}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
  target_sources(${host} PRIVATE "${export_file}" "${force_include_file}")
endfunction()
