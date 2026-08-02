// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Stub translation unit for the onnx_test_data_proto library in onnx-light builds.
// The tml implementation now lives in onnx-light (onnx_light/onnx_proto/tml.h)
// and is header-only; this file exists only to satisfy CMake's static library
// requirement and to validate that the header is reachable.
#if defined(ORT_USE_ONNX_LIGHT)
#include <onnx_light/onnx_proto/tml.h>
#endif
