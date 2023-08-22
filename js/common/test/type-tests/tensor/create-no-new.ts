// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as ort from 'onnxruntime-common';

// calling ort.Tensor() - creating tensor without 'new' is not allowed.
//
// "Value of type 'TensorConstructor & TensorFactory' is not callable. Did you mean to include 'new'?"
//
// {type-tests}|fail|1|2348
ort.Tensor('float32', [1, 2, 3, 4], [2, 2]);
