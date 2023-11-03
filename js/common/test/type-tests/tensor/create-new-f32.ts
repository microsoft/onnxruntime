// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TypedTensor} from 'onnxruntime-common';

// construct from type, data (number array) and shape (number array)
//
// {type-tests}|pass
new Tensor('float32', [1, 2, 3, 4], [2, 2]) as TypedTensor<'float32'>;

// construct from type and data (number array)
//
// {type-tests}|pass
new Tensor('float32', [1, 2, 3, 4]) as TypedTensor<'float32'>;

// construct from type, data (Float32Array) and shape (number array)
//
// {type-tests}|pass
new Tensor('float32', new Float32Array([1, 2, 3, 4]), [2, 2]) as TypedTensor<'float32'>;

// construct from type and data (Float32Array)
//
// {type-tests}|pass
new Tensor('float32', new Float32Array([1, 2, 3, 4])) as TypedTensor<'float32'>;

// construct from data (Float32Array)
//
// {type-tests}|pass
new Tensor(new Float32Array([1, 2, 3, 4])) as TypedTensor<'float32'>;

// construct from data (Float32Array) and shape (number array)
//
// {type-tests}|pass
new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]) as TypedTensor<'float32'>;

// construct (no params) - need params
//
// "Expected 1-3 arguments, but got 0."
//
// {type-tests}|fail|1|2554
new Tensor();

// construct from type - need data
//
// "No overload matches this call."
//
// {type-tests}|fail|1|2769
new Tensor('float32');

// construct from data (number array) - type cannot be inferred.
//
// "No overload matches this call."
//
// {type-tests}|fail|1|2769
new Tensor([1, 2, 3, 4]);

// construct from data (Float32Array) and shape (number) - type mismatch.
//
// "No overload matches this call."
//
// {type-tests}|fail|1|2769
new Tensor(new Float32Array([1, 2, 3, 4]), 2);
