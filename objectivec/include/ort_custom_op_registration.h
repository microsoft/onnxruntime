// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/** C API type forward declaration. */
struct OrtStatus;

/** C API type forward declaration. */
struct OrtApiBase;

/** C API type forward declaration. */
struct OrtSessionOptions;

/**
 * Pointer to a custom op registration function that uses the ONNX Runtime C API.
 *
 * This is a low-level type intended for interoperating with libraries which provide such a function for custom op
 * registration, such as [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions).
 */
typedef struct OrtStatus* (*ORTCAPIRegisterCustomOpsFnPtr)(struct OrtSessionOptions* /*options*/,
                                                           const struct OrtApiBase* /*api*/);
