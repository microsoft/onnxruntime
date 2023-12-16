// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnx.h"
#include "wil/wrl.h"
#include "dx.h"

#if USE_DML
// TODO (pavignol): Revert
// #include <DirectML.h>
#include "core/providers/dml/DirectML2.h"
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#endif USE_DML
