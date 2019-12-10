# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

set(featurizers_URL "https://github.com/microsoft/FeaturizersLibrary.git")
set(featurizers_TAG "acd58fe63baa529e9b318d156ea70d7bf4dc3dad")


set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

# Only due to GIT_CONFIG
if (MSVC)
    ExternalProject_Add(featurizers_lib
            PREFIX ${featurizers_pref}
            GIT_REPOSITORY ${featurizers_URL}
            GIT_TAG ${ngraph_TAG}
            # Need this to properly checkout crlf
            GIT_CONFIG core.autocrlf=input
            SOURCE_DIR ${featurizers_ROOT}
            # Location of CMakeLists.txt
            SOURCE_SUBDIR src/Featurizers
            BINARY_DIR ${featurizers_BINARY_DIR}
            UPDATE_COMMAND ""
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
            UPDATE_COMMAND ""
            INSTALL_COMMAND ""
        )
endif()

add_library(automl_featurizers STATIC IMPORTED)
add_dependencies(automl_featurizers featurizers_lib)
target_include_directories(automl_featurizers INTERFACE ${featurizers_ROOT}/src)
set_property(TARGET automl_featurizers PROPERTY IMPORTED_LOCATION 
  ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/${CMAKE_BUILD_TYPE}/FeaturizersCode.lib)


if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(automl_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
