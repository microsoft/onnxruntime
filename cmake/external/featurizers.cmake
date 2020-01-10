# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

set(featurizers_URL "https://github.com/microsoft/FeaturizersLibrary.git")
set(featurizers_TAG "4948801a488d7c97ba21ca5c07ccc2977c3838af")

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

# Windows required git config setting
# and an a switch whether we link to MSVCRT statically or dynamically
if (WIN32)
    ExternalProject_Add(featurizers_lib
            PREFIX ${featurizers_pref}
            GIT_REPOSITORY ${featurizers_URL}
            GIT_TAG ${featurizers_TAG}
            # Need this to properly checkout crlf
            GIT_CONFIG core.autocrlf=input
            SOURCE_DIR ${featurizers_ROOT}
            # Location of CMakeLists.txt
            SOURCE_SUBDIR src/Featurizers
            BINARY_DIR ${featurizers_BINARY_DIR}
            CMAKE_ARGS -Dfeaturizers_MSVC_STATIC_RUNTIME=${onnxruntime_MSVC_STATIC_RUNTIME}
            INSTALL_COMMAND ""
            
        )
else()
    ExternalProject_Add(featurizers_lib
            PREFIX ${featurizers_pref}
            GIT_REPOSITORY ${featurizers_URL}
            GIT_TAG ${featurizers_TAG}
            SOURCE_DIR ${featurizers_ROOT}
            # Location of CMakeLists.txt
            SOURCE_SUBDIR src/Featurizers
            BINARY_DIR ${featurizers_BINARY_DIR}
            CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            INSTALL_COMMAND ""
        )
endif()

add_library(onnxruntime_featurizers STATIC IMPORTED)
add_dependencies(onnxruntime_featurizers featurizers_lib)
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
