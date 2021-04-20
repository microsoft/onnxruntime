// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from './attribute';
import {InferenceHandler} from './backend';
import {Graph} from './graph';
import {Tensor} from './tensor';

export interface Operator {
  initialize(attributes: Attribute, node: Graph.Node, graph: Graph): void;
  checkInputs(inputs: Tensor[]): boolean;
  run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;
}

export const NUMBER_TYPES: readonly Tensor.DataType[] =
    ['float32', 'float64', 'int32', 'int16', 'int8', 'uint16', 'uint32', 'uint8'];
export const INT_TYPES: readonly Tensor.DataType[] = ['int32', 'int16', 'int8', 'uint16', 'uint32', 'uint8'];
export const FLOAT_TYPES: readonly Tensor.DataType[] = ['float32', 'float64'];
