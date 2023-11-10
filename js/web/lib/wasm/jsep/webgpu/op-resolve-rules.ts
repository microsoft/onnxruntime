// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as binaryOps from './ops/binary-op';
import {concat, parseConcatAttributes} from './ops/concat';
import {conv, parseConvAttributes} from './ops/conv';
import {convTranspose, parseConvTransposeAttributes} from './ops/conv-transpose';
import {expand} from './ops/expand';
import {gemm, parseGemmAttributes} from './ops/gemm';
import {matMul} from './ops/matmul';
import * as pool from './ops/pool';
import {parseReduceAttributes, reduceL1, reduceL2, reduceLogSum, reduceLogSumExp, reduceMax, reduceMean, reduceMin, reduceProd, reduceSum, reduceSumSquare} from './ops/reduce';
import {parseSliceAttributes, slice} from './ops/slice';
import {parseSplitAttributes, split} from './ops/split';
import {parseTransposeAttributes, transpose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {ComputeContext} from './types';

export type RunFunction = (context: ComputeContext, attribute?: unknown) => void;
export type ParseAttributeFunction = (attributeRaw: unknown) => unknown;
export type OperatorImplementation = [RunFunction]|[RunFunction, ParseAttributeFunction];

export const WEBGPU_OP_RESOLVE_RULES: Map<string, OperatorImplementation> = new Map([
  ['Abs', [unaryOps.abs]],
  ['Acos', [unaryOps.acos]],
  ['Acosh', [unaryOps.acosh]],
  ['Add', [binaryOps.add]],
  ['Asin', [unaryOps.asin]],
  ['Asinh', [unaryOps.asinh]],
  ['Atan', [unaryOps.atan]],
  ['Atanh', [unaryOps.atanh]],
  // TODO: support new attributes for AveragePool-10
  ['AveragePool', [pool.averagePool, pool.parseAveragePoolAttributes]],
  ['Ceil', [unaryOps.ceil]],
  ['ClipV10', [unaryOps.clipV10]],
  ['Clip', [unaryOps.clip]],
  ['Concat', [concat, parseConcatAttributes]],
  ['Conv', [conv, parseConvAttributes]],
  ['ConvTranspose', [convTranspose, parseConvTransposeAttributes]],
  ['Cos', [unaryOps.cos]],
  ['Cosh', [unaryOps.cosh]],
  ['Div', [binaryOps.div]],
  ['Elu', [unaryOps.elu, unaryOps.parseAlphaAttributes]],
  ['Erf', [unaryOps.erf]],
  ['Exp', [unaryOps.exp]],
  ['Expand', [expand]],
  ['Floor', [unaryOps.floor]],
  ['Gemm', [gemm, parseGemmAttributes]],
  ['GlobalAveragePool', [pool.globalAveragePool, pool.parseGlobalAveragePoolAttributes]],
  ['GlobalMaxPool', [pool.globalMaxPool, pool.parseGlobalMaxPoolAttributes]],
  ['LeakyRelu', [unaryOps.leakyRelu, unaryOps.parseAlphaAttributes]],
  ['MatMul', [matMul]],
  // TODO: support new attributes for MaxPool-8 and MaxPool-10
  ['MaxPool', [pool.maxPool, pool.parseMaxPoolAttributes]],
  ['Mul', [binaryOps.mul]],
  ['Neg', [unaryOps.neg]],
  ['Pow', [binaryOps.pow]],
  ['Reciprocal', [unaryOps.reciprocal]],
  ['ReduceMin', [reduceMin, parseReduceAttributes]],
  ['ReduceMean', [reduceMean, parseReduceAttributes]],
  ['ReduceMax', [reduceMax, parseReduceAttributes]],
  ['ReduceSum', [reduceSum, parseReduceAttributes]],
  ['ReduceProd', [reduceProd, parseReduceAttributes]],
  ['ReduceL1', [reduceL1, parseReduceAttributes]],
  ['ReduceL2', [reduceL2, parseReduceAttributes]],
  ['ReduceLogSum', [reduceLogSum, parseReduceAttributes]],
  ['ReduceLogSumExp', [reduceLogSumExp, parseReduceAttributes]],
  ['ReduceSumSquare', [reduceSumSquare, parseReduceAttributes]],
  ['Relu', [unaryOps.relu]],
  ['Sigmoid', [unaryOps.sigmoid]],
  ['Sin', [unaryOps.sin]],
  ['Sinh', [unaryOps.sinh]],
  ['Slice', [slice, parseSliceAttributes]],
  ['Split', [split, parseSplitAttributes]],
  ['Sqrt', [unaryOps.sqrt]],
  ['Sub', [binaryOps.sub]],
  ['Tan', [unaryOps.tan]],
  ['Tanh', [unaryOps.tanh]],
  ['ThresholdedRelu', [unaryOps.thresholdedRelu, unaryOps.parseAlphaAttributes]],
  ['Transpose', [transpose, parseTransposeAttributes]],
]);
