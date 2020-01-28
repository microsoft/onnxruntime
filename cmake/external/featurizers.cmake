# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

add_subdirectory(external/FeaturizersLibrary/src/Featurizers ${featurizers_BINARY_DIR} EXCLUDE_FROM_ALL)
set_target_properties(FeaturizersCode PROPERTIES FOLDER "External/FeaturizersLibrary")

add_library(onnxruntime_featurizers STATIC IMPORTED)
add_dependencies(onnxruntime_featurizers FeaturizersCode)

target_include_directories(onnxruntime_featurizers INTERFACE ${featurizers_ROOT}/src)
if(MSVC)
  set_property(TARGET onnxruntime_featurizers PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/${CMAKE_BUILD_TYPE}/FeaturizersCode.lib)
else()
  set_property(TARGET onnxruntime_featurizers PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/libFeaturizersCode.a)
endif()


if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
