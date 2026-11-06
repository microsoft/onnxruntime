# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

foreach(retired_option IN ITEMS
    onnxruntime_ENABLE_TRAINING
    onnxruntime_ENABLE_TRAINING_APIS
    onnxruntime_ENABLE_TRAINING_OPS
    onnxruntime_ENABLE_TRAINING_E2E_TESTS
    onnxruntime_ENABLE_TRAINING_TORCH_INTEROP
    onnxruntime_ENABLE_LAZY_TENSOR
    onnxruntime_ENABLE_EAGER_MODE
    onnxruntime_ENABLE_TRITON
    onnxruntime_BUILD_OPSCHEMA_LIB
    onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  if(${retired_option})
    message(FATAL_ERROR
      "ORT training has been retired. ${retired_option} is no longer supported. "
      "Remove this option or set it to OFF to build ONNX Runtime for inference.")
  endif()
endforeach()
unset(retired_option)
