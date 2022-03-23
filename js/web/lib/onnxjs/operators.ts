// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from './backend';
import {Graph} from './graph';
import {Tensor} from './tensor';

export type OperatorImplementation<ContextType, ReturnType extends Tensor[]|Promise<Tensor[]> = Tensor[]> =
    (inferenceHandler: InferenceHandler, inputs: Tensor[], context: ContextType) => ReturnType;
export type OperatorAsyncImplementation<T> = OperatorImplementation<T, Promise<Tensor[]>>;
export type OperatorInitialization<T> = (node: Graph.Node, graph: Graph) => T;

export interface Operator {
  readonly impl: OperatorImplementation<unknown>|OperatorAsyncImplementation<unknown>;
  readonly context: Graph.Node|unknown;
}

export const NUMBER_TYPES: readonly Tensor.DataType[] =
    ['float32', 'float64', 'int32', 'int16', 'int8', 'uint16', 'uint32', 'uint8'];
export const INT_TYPES: readonly Tensor.DataType[] = ['int32', 'int16', 'int8', 'uint16', 'uint32', 'uint8'];
export const FLOAT_TYPES: readonly Tensor.DataType[] = ['float32', 'float64'];
