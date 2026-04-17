# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#
# Minimum required language standard versions.
#

set(onnxruntime_MINIMUM_C_STANDARD_VERSION 99)
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

function(onnxruntime_ensure_minimum_language_standard_version
         language_standard_version_var_name
         minimum_version)
  if(DEFINED ${language_standard_version_var_name})
    if(${language_standard_version_var_name} VERSION_LESS "${minimum_version}")
      message(FATAL_ERROR "${language_standard_version_var_name} must be at least ${minimum_version}. "
                          "It is ${${language_standard_version_var_name}}.")
    endif()
  else()
    message(STATUS "Setting ${language_standard_version_var_name} to ${minimum_version}")
    set(${language_standard_version_var_name} ${minimum_version} PARENT_SCOPE)
  endif()
endfunction()

onnxruntime_ensure_minimum_language_standard_version(
    CMAKE_C_STANDARD ${onnxruntime_MINIMUM_C_STANDARD_VERSION})

onnxruntime_ensure_minimum_language_standard_version(
    CMAKE_CXX_STANDARD ${onnxruntime_MINIMUM_CXX_STANDARD_VERSION})

#
# Define onnxruntime_<lang>_standard interface targets requiring the minimum standard version.
# These should be used by all onnxruntime targets.
#

add_library(onnxruntime_c_standard INTERFACE)
target_compile_features(onnxruntime_c_standard INTERFACE
                        c_std_${onnxruntime_MINIMUM_C_STANDARD_VERSION})

add_library(onnxruntime_cxx_standard INTERFACE)
target_compile_features(onnxruntime_cxx_standard INTERFACE
                        cxx_std_${onnxruntime_MINIMUM_CXX_STANDARD_VERSION})
