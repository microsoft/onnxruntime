# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

set(featurizers_URL "https://github.com/microsoft/FeaturizersLibrary.git")
set(featurizers_TAG "006df6bff45dac59d378609fe85f6736a901ee93")

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})
set(featurizers_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref})

# Only due to GIT_CONFIG
# Uncoment UPDATE_COMMAND if you work locally
# on the featurizers so cmake does not undo your changes.
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
#            UPDATE_COMMAND ""
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
#            UPDATE_COMMAND ""
            INSTALL_COMMAND ""
        )
endif()

add_library(automl_featurizers STATIC IMPORTED)
add_dependencies(automl_featurizers featurizers_lib)
target_include_directories(automl_featurizers INTERFACE ${featurizers_ROOT}/src)

if(MSVC)
  set_property(TARGET automl_featurizers PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/${CMAKE_BUILD_TYPE}/FeaturizersCode.lib)
else()
  set_property(TARGET automl_featurizers PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_BINARY_DIR}/external/${featurizers_pref}/libFeaturizersCode.a)
endif()

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(automl_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

# Build this in CentOS
# foreach(_test_name IN ITEMS
    # CatImputerFeaturizer_UnitTests
    # DateTimeFeaturizer_UnitTests
    # HashOneHotVectorizerFeaturizer_UnitTests
    # ImputationMarkerFeaturizer_UnitTests
    # LabelEncoderFeaturizer_UnitTests
    # MaxAbsScalarFeaturizer_UnitTests
    # MinMaxScalarFeaturizer_UnitTests
    # MissingDummiesFeaturizer_UnitTests
    # OneHotEncoderFeaturizer_UnitTests
    # RobustScalarFeaturizer_UnitTests
    # SampleAddFeaturizer_UnitTest
    # StringFeaturizer_UnitTest
    # Structs_UnitTest
    # TimeSeriesImputerFeaturizer_UnitTest
# )
    # add_executable(${_test_name} ${featurizers_ROOT}/src/Featurizers/UnitTests/${_test_name}.cpp)
    # add_dependencies(${_test_name} automl_featurizers)
    # target_include_directories(${_test_name} PRIVATE ${featurizers_ROOT}/src)
    # target_link_libraries(${_test_name} automl_featurizers)
    # list(APPEND featurizers_TEST_SRC ${featurizers_ROOT}/src/Featurizers/UnitTests/${_test_name}.cpp)
# endforeach()

# source_group(TREE ${featurizers_ROOT}/src/Featurizers/UnitTests FILES ${featurizers_TEST_SRC})
