// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ORT C API (onnxruntime_c_api.h) type forward declarations.
struct OrtStatus;
struct OrtApiBase;
struct OrtSessionOptions;

/**
 * Custom op registration C function pointer.
 *
 * This is a low-level type intended for interoperating with libraries which provide such a C function for custom op
 * registration, such as onnxruntime-extensions.
 */
typedef OrtStatus* (*ORTCAPIRegisterCustomOpsFnPtr)(OrtSessionOptions* /*options*/, const OrtApiBase* /*api*/);
