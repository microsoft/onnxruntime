// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TypedTensor} from 'onnxruntime-common';

// construct from type, data (string array) and shape (number array)
//
// {type-tests}|pass
new Tensor('string', ['a', 'b', 'c', 'd'], [2, 2]) as TypedTensor<'string'>;

// construct from type and data (string array)
//
// {type-tests}|pass
new Tensor('string', ['a', 'b', 'c', 'd']) as TypedTensor<'string'>;

// construct from data (string array)
//
// {type-tests}|pass
new Tensor(['a', 'b', 'c', 'd']) as TypedTensor<'string'>;
