# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

file(GLOB_RECURSE onnxruntime_automl_featurizers_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/FeaturizerPrep/*.h"
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/FeaturizerPrep/Featurizers/*.h"
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/FeaturizerPrep/Featurizers/*.cpp"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_automl_featurizers_srcs})

add_library(onnxruntime_automl_featurizers ${onnxruntime_automl_featurizers_srcs})

target_include_directories(onnxruntime_automl_featurizers PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(onnxruntime_automl_featurizers PROPERTIES FOLDER "ONNXRuntime")

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_automl_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
