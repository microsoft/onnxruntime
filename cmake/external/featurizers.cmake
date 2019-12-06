# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently


#set(featurizers_URL "https://github.com/microsoft/FeaturizersLibrary.git")
#set(featurizers_TAG "31c78493d7a6a91399aef5fbfd87d6d02fd19852")

set(featurizers_pref FeaturizersLibrary)
set(featurizers_ROOT ${PROJECT_SOURCE_DIR}/external/${featurizers_pref})

# Need this to properly checkout crlf
# if (MSVC)
    # ExternalProject_Add(featurizers_lib_download
            # PREFIX ${featurizers_pref}
            # GIT_REPOSITORY ${featurizers_URL}
            # GIT_TAG ${ngraph_TAG}
            # GIT_CONFIG core.autocrlf=input
            # SOURCE_DIR ${featurizers_ROOT}
            # CONFIGURE_COMMAND ""
            # BUILD_COMMAND ""
            # UPDATE_COMMAND ""
            # INSTALL_COMMAND ""
        # )
# else()
    # ExternalProject_Add(featurizers_lib_download
            # PREFIX ${featurizers_pref}
            # GIT_REPOSITORY ${featurizers_URL}
            # GIT_TAG ${featurizers_TAG}
            # SOURCE_DIR ${featurizers_ROOT}
            # CONFIGURE_COMMAND ""
            # BUILD_COMMAND ""
            # UPDATE_COMMAND ""
            # INSTALL_COMMAND ""
        # )
# endif()


set(automl_featurizers_srcs 
  "${featurizers_ROOT}/src/Featurizers/CatImputerFeaturizer.h"
  "${featurizers_ROOT}/src/Featurizers/DateTimeFeaturizer.h"
  "${featurizers_ROOT}/src/Featurizers/SampleAddFeaturizer.h"
  "${featurizers_ROOT}/src/Featurizers/StringFeaturizer.h"
  "${featurizers_ROOT}/src/Featurizers/TimeSeriesImputerFeaturizer.h"

  "${featurizers_ROOT}/src/Featurizers/Components/Components.h"
  "${featurizers_ROOT}/src/Featurizers/Components/InferenceOnlyFeaturizerImpl.h"
  "${featurizers_ROOT}/src/Featurizers/Components/PipelineExecutionEstimatorImpl.h"
  "${featurizers_ROOT}/src/Featurizers/Components/TimeSeriesFrequencyEstimator.h"
  "${featurizers_ROOT}/src/Featurizers/Components/TimeSeriesImputerTransformer.h"
  "${featurizers_ROOT}/src/Featurizers/Components/TrainingOnlyEstimatorImpl.h"

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
