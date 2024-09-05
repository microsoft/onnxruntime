// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as ort from 'onnxruntime-common';

// construct from Uint8ClampedArray
//
// {type-tests}|pass
new ort.Tensor(new Uint8ClampedArray(1))


