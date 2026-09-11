# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#
# Minimum required language standard versions.
#

set(onnxruntime_MINIMUM_C_STANDARD_VERSION   99)
set(onnxruntime_MINIMUM_CXX_STANDARD_VERSION 20)

#
# Handle CMAKE_<LANG>_STANDARD variables.
#
# We only set them if unset. Otherwise, enforce a minimum value.
#
# We care about the CMAKE_<LANG>_STANDARD variables because we typically want our dependencies to be built with the
# same language standard versions as the rest of our code.
# E.g., this is important for Abseil.
#

# "Normalize" means make suitable for comparison.
function(onnxruntime_normalize_language_standard_version
         language_standard_version_var_name
         version
         normalized_version_var_name)
  set(normalized_version ${version})

  # Note: For CMAKE_C_STANDARD and CMAKE_CXX_STANDARD, we assume two-digit versions based on years.
  if("${language_standard_version_var_name}" STREQUAL "CMAKE_C_STANDARD")
    if("${version}" EQUAL "90" OR "${version}" EQUAL "99")
      set(base_year 1900)
    else()
      set(base_year 2000)
    endif()
    math(EXPR normalized_version "${base_year} + ${version}")
  elseif("${language_standard_version_var_name}" STREQUAL "CMAKE_CXX_STANDARD")
    if("${version}" EQUAL "98")
      set(base_year 1900)
    else()
      set(base_year 2000)
    endif()
    math(EXPR normalized_version "${base_year} + ${version}")
  endif()

  set(${normalized_version_var_name} ${normalized_version} PARENT_SCOPE)
endfunction()

function(onnxruntime_ensure_minimum_language_standard_version
         language_standard_version_var_name
         minimum_version)
  if(DEFINED ${language_standard_version_var_name})
    onnxruntime_normalize_language_standard_version(
        ${language_standard_version_var_name} "${minimum_version}" normalized_minimum_version)
    onnxruntime_normalize_language_standard_version(
        ${language_standard_version_var_name} "${${language_standard_version_var_name}}" normalized_version)

    if(normalized_version VERSION_LESS normalized_minimum_version)
      message(FATAL_ERROR "${language_standard_version_var_name} must be at least ${minimum_version}. "
                          "It is ${${language_standard_version_var_name}}.")
    endif()
  else()
    message(STATUS "Setting ${language_standard_version_var_name} to ${minimum_version}")
    set(${language_standard_version_var_name} ${minimum_version} PARENT_SCOPE)
  endif()
endfunction()

onnxruntime_ensure_minimum_language_standard_version(CMAKE_C_STANDARD   ${onnxruntime_MINIMUM_C_STANDARD_VERSION})
onnxruntime_ensure_minimum_language_standard_version(CMAKE_CXX_STANDARD ${onnxruntime_MINIMUM_CXX_STANDARD_VERSION})

#
# Define onnxruntime_<lang>_std_compile_feature variables specifying the <lang> standard version compile feature name.
# These should be used by all onnxruntime targets via target_compile_features().
#

set(onnxruntime_c_std_compile_feature   c_std_${onnxruntime_MINIMUM_C_STANDARD_VERSION})
set(onnxruntime_cxx_std_compile_feature cxx_std_${onnxruntime_MINIMUM_CXX_STANDARD_VERSION})
