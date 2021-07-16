// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from './tensor';

type NonTensorType = never;

/**
 * Type OnnxValue Represents both tensors and non-tensors value for model's inputs/outputs.
 *
 * NOTE: currently not support non-tensor
 */
export type OnnxValue = Tensor|NonTensorType;
