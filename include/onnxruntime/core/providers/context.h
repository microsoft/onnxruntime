// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_cxx_api.h>

struct Context {
	Context() = default;
	virtual ~Context() {};
	virtual void Init(const OrtKernelContext&) {};
};