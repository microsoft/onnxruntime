# This file contains some wrappers for cmake's FetchContent functions. The wrappers added the following functionalities:
# 1. Group the VC projects into the "external" folder. We can do it at there in a centralized way instead
#    of doing it one by one.
# 2. Set the cmake property COMPILE_WARNING_AS_ERROR to OFF for these external projects.

function(onnxruntime_fetchcontent_declare contentName)
    FetchContent_Declare(${ARGV})
    string(TOLOWER ${contentName} contentNameLower)
    list(FIND ARGN SOURCE_SUBDIR index_SOURCE_SUBDIR)
    if(index_SOURCE_SUBDIR GREATER_EQUAL 0)
      cmake_parse_arguments(PARSE_ARGV 1 ARG "" "SOURCE_SUBDIR" "") 
      set(onnxruntime_${contentNameLower}_cmake_src_dir "${ARG_SOURCE_SUBDIR}" PARENT_SCOPE)
    endif()
endfunction()

macro(onnxruntime_fetchcontent_makeavailable)
    set(ONNXRUNTIME_CMAKE_SKIP_INSTALL_RULES_OLD_VALUE
      "${CMAKE_SKIP_INSTALL_RULES}")
    # If we don't skip the install rules we will hit errors from re2 like:
    # CMake Error: install(EXPORT "re2Targets" ...) includes target "re2" which requires target "absl_base" that is not in any export set.
    set(CMAKE_SKIP_INSTALL_RULES TRUE)
    FetchContent_MakeAvailable(${ARGV})
    foreach(contentName IN ITEMS ${ARGV})
      string(TOLOWER ${contentName} contentNameLower)
      set(content_src_dir  "${${contentNameLower}_SOURCE_DIR}")
      if(NOT "${onnxruntime_${contentNameLower}_cmake_src_dir}" STREQUAL "")
        string(APPEND content_src_dir "/${onnxruntime_${contentNameLower}_cmake_src_dir}")
      endif()
      get_property(subdir_import_targets DIRECTORY "${content_src_dir}" PROPERTY BUILDSYSTEM_TARGETS)
      foreach(subdir_target ${subdir_import_targets})
          if(TARGET ${subdir_target})
              get_target_property(subdir_target_type ${subdir_target} TYPE)
              if(subdir_target_type STREQUAL "EXECUTABLE")
                get_target_property(subdir_target_osx_arch ${subdir_target} OSX_ARCHITECTURES)
                if (subdir_target_osx_arch)
                  if (NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} IN_LIST subdir_target_osx_arch)
                    message("Added an executable target ${subdir_target} but it can not run natively on ${CMAKE_HOST_SYSTEM_PROCESSOR}, we will try to modify it")
                  endif()
                endif()
              endif()
              set_target_properties(${subdir_target} PROPERTIES FOLDER "External")
              set_target_properties(${subdir_target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
          endif()
      endforeach()
    endforeach()
    set(CMAKE_SKIP_INSTALL_RULES ${ONNXRUNTIME_CMAKE_SKIP_INSTALL_RULES_OLD_VALUE})
endmacro()
