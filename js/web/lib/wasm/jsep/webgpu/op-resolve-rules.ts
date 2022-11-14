// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as binaryOps from './ops/binary-op';
// import {concat, parseConcatAttributes} from './ops/concat';
import {conv, parseConvAttributes} from './ops/conv';
// import {gather, parseGatherAttributes} from './ops/gather';
// import {gemm, parseGemmAttributesV11, parseGemmAttributesV7} from './ops/gemm';
// import {matMul, parseMatMulAttributes} from './ops/matmul';
// import {averagePool, globalAveragePool, globalMaxPool, maxPool, parseAveragePoolAttributes,
// parseGlobalAveragePoolAttributes, parseMaxPoolAttributes} from './ops/pool'; import {sum} from
// './ops/reduce-tensors'; import {reshape} from './ops/reshape'; import {shape} from './ops/shape';
// import {parseSliceAttributes, slice, sliceV10} from './ops/slice';
// import {parseSqueezeAttributes, squeeze, squeezeV13} from './ops/squeeze';
// import {parseTransposeAttributes, transpose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {ComputeContext} from './types';

// import {parseUnsqueezeAttributes, unsqueeze, unsqueezeV13} from './ops/unsqueeze';

export type RunFunction = (context: ComputeContext, attribute?: unknown) => number;
export type ParseAttributeFunction = (attributeRaw: unknown) => unknown;
export type OperatorImplementation = [RunFunction]|[RunFunction, ParseAttributeFunction];

export const WEBGPU_OP_RESOLVE_RULES: Map<string, OperatorImplementation> = new Map([
  ['Abs', [unaryOps.abs]], ['Acos', [unaryOps.acos]], ['Acosh', [unaryOps.acosh]], ['Add', [binaryOps.add]],
  // ['And', '', '7+', binaryOps.and],
  ['Asin', [unaryOps.asin]], ['Asinh', [unaryOps.asinh]], ['Atan', [unaryOps.atan]], ['Atanh', [unaryOps.atanh]],
  // TODO: support new attributes for AveragePool-10
  //['AveragePool', '', '7+', averagePool, parseAveragePoolAttributes],
  // ['BatchNormalization', '', '7+', batchNormalization, parseBatchNormalizationAttributes],
  // ['Cast', '', '6+', cast, parseCastAttributes],
  ['Ceil', [unaryOps.ceil]],
  // ['Clip', '', '6-10', unaryOps.clip, unaryOps.parseClipAttributes],
  //['Clip', '', '11+', unaryOps.clipV11], ['Concat', '', '4+', concat, parseConcatAttributes],
  ['Conv', [conv, parseConvAttributes]], ['Cos', [unaryOps.cos]], ['Cosh', [unaryOps.cosh]], ['Div', [binaryOps.div]],
  // ['Dropout', '', '7+', unaryOps.identity],
  // ['DepthToSpace', '', '1+', depthToSpace, parseDepthToSpaceAttributes],
  // ['Equal', '', '7+', binaryOps.equal],
  ['Elu', [unaryOps.elu, unaryOps.parseEluAttributes]],  //['Exp', [unaryOps.exp]],
  // ['Flatten', '', '1+', flatten, parseFlattenAttributes],
  ['Floor', [unaryOps.floor]],
  // ['FusedConv', 'com.microsoft', '1+', conv, parseConvAttributes],
  //['Gather', '', '1+', gather, parseGatherAttributes], ['Gemm', '', '7-10', gemm, parseGemmAttributesV7],
  //['Gemm', '', '11+', gemm, parseGemmAttributesV11],
  //['GlobalAveragePool', '', '1+', globalAveragePool, parseGlobalAveragePoolAttributes],
  //['GlobalMaxPool', '', '1+', globalMaxPool],
  // ['Greater', '', '7+', binaryOps.greater],
  // ['Identity', '', '1+', unaryOps.identity],
  // ['ImageScaler', '', '1+', imageScaler, parseImageScalerAttributes],
  // ['InstanceNormalization', '', '6+', instanceNormalization, parseInstanceNormalizationAttributes],
  //['LeakyRelu', '', '6+', unaryOps.leakyRelu, unaryOps.parseLeakyReluAttributes],
  // ['Less', '', '7+', binaryOps.less],
  //['Log', '', '6+', unaryOps.log], ['MatMul', '', '1+', matMul, parseMatMulAttributes],
  // TODO: support new attributes for MaxPool-8 and MaxPool-10
  //['MaxPool', '', '1+', maxPool, parseMaxPoolAttributes],
  ['Mul', [binaryOps.mul]], ['Neg', [unaryOps.neg]],
  // ['Not', '', '1+', unaryOps.not],
  // ['Or', '', '7+', binaryOps.or],
  // ['Pad', '', '2-10', padV2, parsePadAttributesV2],
  // ['Pad', '', '11+', padV11, parsePadAttributesV11],
  ['Pow', [binaryOps.pow]],
  // ['PRelu', '', '7+', binaryOps.pRelu],
  ['Reciprocal', [unaryOps.reciprocal]],
  // ['ReduceLogSum', '', '1+', reduceLogSum, parseReduceAttributes],
  // ['ReduceMax', '', '1+', reduceMax, parseReduceAttributes],
  // ['ReduceMean', '', '1+', reduceMean, parseReduceAttributes],
  // ['ReduceMin', '', '1+', reduceMin, parseReduceAttributes],
  // ['ReduceProd', '', '1+', reduceProd, parseReduceAttributes],
  // ['ReduceSum', '', '1-12', reduceSum, parseReduceAttributes],
  // ['ReduceSumSquare', '', '1+', reduceLogSumSquare, parseReduceAttributes],
  //['Relu', '', '6+', unaryOps.relu], ['Reshape', '', '5+', reshape],
  // ['Resize', '', '10', resize, parseResizeAttributesV10],
  // ['Resize', '', '11+', resize, parseResizeAttributesV11],
  //['Shape', '', '1+', shape], ['Sigmoid', '', '6+', unaryOps.sigmoid],
  ['Sin', [unaryOps.sin]], ['Sinh', [unaryOps.sinh]],
  //['Slice', '', '10+', sliceV10],  // TODO: support 'steps' for Slice-10
  //['Slice', '', '1-9', slice, parseSliceAttributes],
  // // The "semantic" meaning of axis has changed in opset-13.
  // ['Softmax', '', '1-12', softmax, parseSoftmaxAttributes],
  // ['Softmax', '', '13+', softmaxV13, parseSoftmaxAttributesV13],
  // // 'Split' operator has an optional attribute 'split'
  // // this attribute determines how the specified axis of input data is split.
  // // When the attribute is missing, we need the count of number of outputs
  // // so that we can determine the 'split' attribute from the runtime input to the Operator
  // ['Split', '', '2-12', split, parseSplitAttributes],
  ['Sqrt', [unaryOps.sqrt]],
  // ['Squeeze', '', '1-12', squeeze, parseSqueezeAttributes],
  //['Squeeze', '', '13+', squeezeV13],
  ['Sub', [binaryOps.sub]],  // ['Sum', '', '6+', sum],
  ['Tan', [unaryOps.tan]], ['Tanh', [unaryOps.tanh]],
  // ['Tile', '', '6+', tile],
  //['Transpose', '', '1+', transpose, parseTransposeAttributes],
  // ['Upsample', '', '7-8', upsample, parseUpsampleAttributesV7],
  // ['Upsample', '', '9', upsample, parseUpsampleAttributesV9],
  //['Unsqueeze', '', '1-12', unsqueeze, parseUnsqueezeAttributes], ['Unsqueeze', '', '13+', unsqueezeV13],
  // ['Xor', '', '7+', binaryOps.xor],
]);
