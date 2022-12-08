# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# This file was copied from cmake source with modifications:
# 1. Add the EXCLUDE_FROM_ALL keyword when this function calls add_subdirectory. It will also resolve the
#    'make install' issue.
# 2. Group the VC projects into the "external" folder. We can do it at there in a centralized way instead
#    of doing it one by one.
# 3. Set the cmake property COMPILE_WARNING_AS_ERROR to OFF for these external projects.

macro(onnxruntime_fetchcontent_makeavailable)

  # We must append an item, even if the variable is unset, so prefix its value.
  # We will strip that prefix when we pop the value at the end of the macro.
  list(APPEND __cmake_fcCurrentVarsStack
    "__fcprefix__${CMAKE_VERIFY_INTERFACE_HEADER_SETS}"
  )
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)

  get_property(__cmake_providerCommand GLOBAL PROPERTY
    __FETCHCONTENT_MAKEAVAILABLE_SERIAL_PROVIDER
  )
  foreach(__cmake_contentName IN ITEMS ${ARGV})
    string(TOLOWER ${__cmake_contentName} __cmake_contentNameLower)

    # If user specified FETCHCONTENT_SOURCE_DIR_... for this dependency, that
    # overrides everything else and we shouldn't try to use find_package() or
    # a dependency provider.
    string(TOUPPER ${__cmake_contentName} __cmake_contentNameUpper)
    if("${FETCHCONTENT_SOURCE_DIR_${__cmake_contentNameUpper}}" STREQUAL "")
      # Dependency provider gets first opportunity, but prevent infinite
      # recursion if we are called again for the same thing
      if(NOT "${__cmake_providerCommand}" STREQUAL "" AND
        NOT DEFINED __cmake_fcProvider_${__cmake_contentNameLower})
        message(VERBOSE
          "Trying FETCHCONTENT_MAKEAVAILABLE_SERIAL dependency provider for "
          "${__cmake_contentName}"
        )
        # It's still valid if there are no saved details. The project may have
        # been written to assume a dependency provider is always set and will
        # provide dependencies without having any declared details for them.
        __FetchContent_getSavedDetails(${__cmake_contentName} __cmake_contentDetails)
        set(__cmake_providerArgs
          "FETCHCONTENT_MAKEAVAILABLE_SERIAL"
          "${__cmake_contentName}"
        )
        # Empty arguments must be preserved because of things like
        # GIT_SUBMODULES (see CMP0097)
        foreach(__cmake_item IN LISTS __cmake_contentDetails)
          string(APPEND __cmake_providerArgs " [==[${__cmake_item}]==]")
        endforeach()

        # This property might be defined but empty. As long as it is defined,
        # find_package() can be called.
        get_property(__cmake_addfpargs GLOBAL PROPERTY
          _FetchContent_${contentNameLower}_find_package_args
          DEFINED
        )
        if(__cmake_addfpargs)
          get_property(__cmake_fpargs GLOBAL PROPERTY
            _FetchContent_${contentNameLower}_find_package_args
          )
          string(APPEND __cmake_providerArgs " FIND_PACKAGE_ARGS")
          foreach(__cmake_item IN LISTS __cmake_fpargs)
            string(APPEND __cmake_providerArgs " [==[${__cmake_item}]==]")
          endforeach()
        endif()

        # Calling the provider could lead to onnxruntime_fetchcontent_makeavailable() being
        # called for a nested dependency. That nested call may occur in the
        # current variable scope. We have to save and restore the variables we
        # need preserved.
        list(APPEND __cmake_fcCurrentVarsStack
          ${__cmake_contentName}
          ${__cmake_contentNameLower}
        )

        set(__cmake_fcProvider_${__cmake_contentNameLower} YES)
        cmake_language(EVAL CODE "${__cmake_providerCommand}(${__cmake_providerArgs})")

        list(POP_BACK __cmake_fcCurrentVarsStack
          __cmake_contentNameLower
          __cmake_contentName
        )

        unset(__cmake_fcProvider_${__cmake_contentNameLower})
        unset(__cmake_providerArgs)
        unset(__cmake_addfpargs)
        unset(__cmake_fpargs)
        unset(__cmake_item)
        unset(__cmake_contentDetails)

        FetchContent_GetProperties(${__cmake_contentName})
        if(${__cmake_contentNameLower}_POPULATED)
          continue()
        endif()
      endif()

      # Check if we've been asked to try find_package() first, even if we
      # have already populated this dependency. If we previously tried to
      # use find_package() for this and it succeeded, those things might
      # no longer be in scope, so we have to do it again.
      get_property(__cmake_haveFpArgs GLOBAL PROPERTY
        _FetchContent_${__cmake_contentNameLower}_find_package_args DEFINED
      )
      if(__cmake_haveFpArgs)
        unset(__cmake_haveFpArgs)
        message(VERBOSE "Trying find_package(${__cmake_contentName} ...) before FetchContent")
        get_property(__cmake_fpArgs GLOBAL PROPERTY
          _FetchContent_${__cmake_contentNameLower}_find_package_args
        )

        # This call could lead to onnxruntime_fetchcontent_makeavailable() being called for
        # a nested dependency and it may occur in the current variable scope.
        # We have to save/restore the variables we need to preserve.
        list(APPEND __cmake_fcCurrentNameStack
          ${__cmake_contentName}
          ${__cmake_contentNameLower}
        )
        find_package(${__cmake_contentName} ${__cmake_fpArgs})
        list(POP_BACK __cmake_fcCurrentNameStack
          __cmake_contentNameLower
          __cmake_contentName
        )
        unset(__cmake_fpArgs)

        if(${__cmake_contentName}_FOUND)
          FetchContent_SetPopulated(${__cmake_contentName})
          FetchContent_GetProperties(${__cmake_contentName})
          continue()
        endif()
      endif()
    else()
      unset(__cmake_haveFpArgs)
    endif()

    FetchContent_GetProperties(${__cmake_contentName})
    if(NOT ${__cmake_contentNameLower}_POPULATED)
      FetchContent_Populate(${__cmake_contentName})
      __FetchContent_setupFindPackageRedirection(${__cmake_contentName})

      # Only try to call add_subdirectory() if the populated content
      # can be treated that way. Protecting the call with the check
      # allows this function to be used for projects that just want
      # to ensure the content exists, such as to provide content at
      # a known location. We check the saved details for an optional
      # SOURCE_SUBDIR which can be used in the same way as its meaning
      # for ExternalProject. It won't matter if it was passed through
      # to the ExternalProject sub-build, since it would have been
      # ignored there.
      set(__cmake_srcdir "${${__cmake_contentNameLower}_SOURCE_DIR}")
      __FetchContent_getSavedDetails(${__cmake_contentName} __cmake_contentDetails)
      if("${__cmake_contentDetails}" STREQUAL "")
        message(FATAL_ERROR "No details have been set for content: ${__cmake_contentName}")
      endif()
      cmake_parse_arguments(__cmake_arg "SYSTEM" "SOURCE_SUBDIR" "" ${__cmake_contentDetails})
      if(NOT "${__cmake_arg_SOURCE_SUBDIR}" STREQUAL "")
        string(APPEND __cmake_srcdir "/${__cmake_arg_SOURCE_SUBDIR}")
      endif()

      if(EXISTS ${__cmake_srcdir}/CMakeLists.txt)
          add_subdirectory(${__cmake_srcdir} ${${__cmake_contentNameLower}_BINARY_DIR} EXCLUDE_FROM_ALL)
          get_property(subdir_import_targets DIRECTORY "${__cmake_srcdir}" PROPERTY BUILDSYSTEM_TARGETS)
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
      endif()

      unset(__cmake_srcdir)
      unset(__cmake_contentDetails)
      unset(__cmake_arg_SOURCE_SUBDIR)
    endif()
  endforeach()
  # Prefix will be "__fcprefix__"
  list(POP_BACK __cmake_fcCurrentVarsStack __cmake_original_verify_setting)
  string(SUBSTRING "${__cmake_original_verify_setting}"
    12 -1 __cmake_original_verify_setting
  )
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ${__cmake_original_verify_setting})

  # clear local variables to prevent leaking into the caller's scope
  unset(__cmake_contentName)
  unset(__cmake_contentNameLower)
  unset(__cmake_contentNameUpper)
  unset(__cmake_providerCommand)
  unset(__cmake_original_verify_setting)
endmacro()