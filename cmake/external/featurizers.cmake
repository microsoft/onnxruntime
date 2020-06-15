# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

add_definitions(-DML_FEATURIZERS)

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

add_subdirectory(external/FeaturizersLibrary/src/Featurizers ${featurizers_BINARY_DIR} EXCLUDE_FROM_ALL)
set_target_properties(FeaturizersCode PROPERTIES FOLDER "External/FeaturizersLibrary")
target_include_directories(FeaturizersCode INTERFACE ${featurizers_ROOT}/src)

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(FeaturizersCode PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
if (WINDOWS_STORE)
    # Library requires narrow version of APIs
    target_compile_options(FeaturizersCode PRIVATE "/U UNICODE" "/U _UNICODE")
endif()

add_library(onnxruntime_featurizers ALIAS FeaturizersCode)
