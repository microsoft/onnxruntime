# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(model-explorer model-explorer.cpp)
target_link_libraries(model-explorer PRIVATE onnxruntime)

add_executable(batch-model-explorer batch-model-explorer.cpp)
target_link_libraries(batch-model-explorer PRIVATE onnxruntime)