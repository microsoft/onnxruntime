// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TypedTensor} from 'onnxruntime-common';

// construct from type, data (boolean array) and shape (number array)
//
// {type-tests}|pass
new Tensor('bool', [true, true, false, false], [2, 2]) as TypedTensor<'bool'>;

// construct from type and data (boolean array)
//
// {type-tests}|pass
new Tensor('bool', [true, true, false, false]) as TypedTensor<'bool'>;

// construct from type, data (Uint8Array) and shape (number array)
//
// {type-tests}|pass
new Tensor('bool', new Uint8Array([1, 1, 0, 0]), [2, 2]) as TypedTensor<'bool'>;

// construct from type and data (Uint8Array)
//
// {type-tests}|pass
new Tensor('bool', new Uint8Array([1, 1, 0, 0])) as TypedTensor<'bool'>;

// construct from data (boolean array)
//
// {type-tests}|pass
new Tensor([true, true, false, false]) as TypedTensor<'bool'>;

// construct from data (Uint8Array) - type is inferred as 'uint8'
//
// "Conversion of type 'TypedTensor<"uint8">' to type 'TypedTensor<"bool">' may be a mistake because ..."
//
// {type-tests}|fail|1|2352
new Tensor(new Uint8Array([1, 1, 0, 0])) as TypedTensor<'bool'>;
