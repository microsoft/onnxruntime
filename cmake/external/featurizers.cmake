# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently


set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})

file(GLOB automl_featurizers_srcs CONFIGURE_DEPENDS
  "${featurizers_ROOT}/src/Featurizers/*.h"
  "${featurizers_ROOT}/src/Featurizers/Components/*.h"
  "${featurizers_ROOT}/src/Featurizers/DateTimeFeaturizer.cpp"
  "${featurizers_ROOT}/src/Featurizers/SampleAddFeaturizer.cpp"
  )

add_library(automl_featurizers STATIC ${automl_featurizers_srcs})
target_include_directories(automl_featurizers PUBLIC ${featurizers_ROOT}/src)

source_group(TREE ${featurizers_ROOT} FILES ${automl_featurizers_srcs})
set_target_properties(automl_featurizers PROPERTIES FOLDER "AutoMLFeaturizers")

#target_include_directories(automl_featurizers PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(automl_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
