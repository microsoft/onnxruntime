# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

add_definitions(-DML_FEATURIZERS)

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

add_subdirectory(external/FeaturizersLibrary/src/Featurizers ${featurizers_BINARY_DIR} EXCLUDE_FROM_ALL)

set_target_properties(FeaturizersCode PROPERTIES FOLDER "External/FeaturizersLibrary")
target_include_directories(FeaturizersCode PUBLIC ${featurizers_ROOT}/src)

add_library(onnxruntime_featurizers ALIAS FeaturizersCode)





#add_library(onnxruntime_featurizers_comp STATIC IMPORTED)
#add_dependencies(onnxruntime_featurizers_comp FeaturizersCode)


# target_include_directories(onnxruntime_featurizers INTERFACE ${featurizers_ROOT}/src)
# BugBug if(MSVC)
# BugBug   set_property(TARGET onnxruntime_featurizers PROPERTY IMPORTED_LOCATION
# BugBug     ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/${CMAKE_BUILD_TYPE}/FeaturizersCode.lib)
# BugBug   set_property(TARGET onnxruntime_featurizers_comp PROPERTY IMPORTED_LOCATION
# BugBug     ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/${CMAKE_BUILD_TYPE}/FeaturizersComponentsCode.lib)
# BugBug else()
# BugBug   set_property(TARGET onnxruntime_featurizers PROPERTY IMPORTED_LOCATION
# BugBug     ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/libFeaturizersCode.a)
# BugBug   set_property(TARGET onnxruntime_featurizers_comp PROPERTY IMPORTED_LOCATION
# BugBug     ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/libFeaturizersComponentsCode.a)
# BugBug endif()
# BugBug
# BugBug
# BugBug if (WIN32)
# BugBug     # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
# BugBug     set_target_properties(onnxruntime_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
# BugBug endif()
# BugBug
