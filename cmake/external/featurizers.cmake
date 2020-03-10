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
if(WINDOWS_STORE)
    # protobuf and FeaturizersLibrary use Win32 desktop APIs; this must be fixed!
    # See https://dev.azure.com/onnxruntime/2a773b67-e88b-4c7f-9fc0-87d31fea8ef2/_apis/build/builds/119296/logs/25
    # target_compile_options(libprotoc PRIVATE "-DWINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP")
    target_compile_options(FeaturizersCode PRIVATE "/FI ${CMAKE_CURRENT_SOURCE_DIR}\\set_winapi_family.h")
    target_link_libraries(FeaturizersCode kernel32.lib)
endif()

add_library(onnxruntime_featurizers ALIAS FeaturizersCode)
