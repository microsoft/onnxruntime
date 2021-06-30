// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OpSet} from '../../opset';

import {batchNormalization, parseBatchNormalizationAttributes} from './ops/batch-normalization';
import * as binaryOps from './ops/binary-op';
// import {WebGLClip} from './ops/clip';
// import {WebGLConcat} from './ops/concat';
// import {WebGLConv} from './ops/conv';
// import {WebGLDepthToSpace} from './ops/depth-to-space';
// import {WebGLDropout} from './ops/dropout';
// import {WebGLElu} from './ops/elu';
// import {WebGLFlatten} from './ops/flatten';
// import {WebGLGather} from './ops/gather';
// import {WebGLGemm} from './ops/gemm';
// import {WebGLImageScaler} from './ops/image-scaler';
// import {WebGLInstanceNormalization} from './ops/instance-normalization';
// import {WebGLLeakyRelu} from './ops/leaky-relu';
// import {WebGLMatMul} from './ops/matmul';
// import {WebGLPad} from './ops/pad';
// import {WebGLAveragePool, WebGLGlobalAveragePool, WebGLGlobalMaxPool, WebGLMaxPool} from './ops/pool';
// import * as reduceOps from './ops/reduce';
import {reshape} from './ops/reshape';
// import {WebGLResizePacked} from './ops/resize-packed';
// import {WebGLShape} from './ops/shape';
// import {WebGLSlice, WebGLSliceV10} from './ops/slice';
// import {WebGLSoftmax} from './ops/softmax';
// import {WebGLSplit} from './ops/split';
// import {WebGLSqueeze} from './ops/squeeze';
// import {WebGLSum} from './ops/sum';
// import {WebGLTile} from './ops/tile';
// import {WebGLTranspose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
// import {WebGLUnsqueeze} from './ops/unsqueeze';
// import {WebGLUpsample} from './ops/upsample';

export const WEBGL_OP_RESOLVE_RULES: readonly OpSet.ResolveRule[] = [
  ['Abs', '', '6+', unaryOps.abs],
  ['Acos', '', '7+', unaryOps.acos],
  ['Add', '', '7+', binaryOps.add],
  ['And', '', '7+', binaryOps.and],
  ['Asin', '', '7+', unaryOps.asin],
  ['Atan', '', '7+', unaryOps.atan],

  // ['AveragePool', '', '7-10', () => new WebGLAveragePool()],  // TODO: support new attributes for AveragePool-10
  ['BatchNormalization', '', '7+', batchNormalization, parseBatchNormalizationAttributes],
  ['Ceil', '', '6+', unaryOps.ceil],
  // ['Clip', '', '6-10', () => new WebGLClip()],
  // ['Concat', '', '4+', () => new WebGLConcat()],
  // ['Conv', '', '1+', () => new WebGLConv()],
  ['Cos', '', '7+', unaryOps.cos],
  ['Div', '', '7+', binaryOps.div],
  // ['Dropout', '', '7+', () => new WebGLDropout()],
  // ['DepthToSpace', '', '1+', () => new WebGLDepthToSpace()],
  ['Equal', '', '7+', binaryOps.equal],
  // ['Elu', '', '6+', () => new WebGLElu()],
  ['Exp', '', '6+', unaryOps.exp],
  // ['Flatten', '', '1+', () => new WebGLFlatten()],
  ['Floor', '', '6+', unaryOps.floor],
  // ['Gather', '', '1+', () => new WebGLGather()],
  // ['Gemm', '', '7-10', () => new WebGLGemm(false)],
  // ['Gemm', '', '11+', () => new WebGLGemm(true)],
  // ['GlobalAveragePool', '', '1+', () => new WebGLGlobalAveragePool()],
  // ['GlobalMaxPool', '', '1+', () => new WebGLGlobalMaxPool()],
  ['Greater', '', '7+', binaryOps.greater],
  ['Identity', '', '1+', unaryOps.identity],
  // ['ImageScaler', '', '1+', () => new WebGLImageScaler()],
  // ['InstanceNormalization', '', '6+', () => new WebGLInstanceNormalization()],
  // ['LeakyRelu', '', '6+', () => new WebGLLeakyRelu()],
  ['Less', '', '7+', binaryOps.less],
  ['Log', '', '6+', unaryOps.log],
  // ['MatMul', '', '1+', () => new WebGLMatMul()],
  // ['MaxPool', '', '1-9', () => new WebGLMaxPool()],  // TODO: support new attributes for MaxPool-8 and MaxPool-10
  ['Mul', '', '7+', binaryOps.mul],
  ['Neg', '', '6+', unaryOps.neg],
  ['Not', '', '1+', unaryOps.not],
  ['Or', '', '7+', binaryOps.or],
  // ['Pad', '', '2-10', () => new WebGLPad()],
  ['Pow', '', '7+', binaryOps.pow],
  ['PRelu', '', '7+', binaryOps.pRelu],
  // ['ReduceLogSum', '', '1+', () => new reduceOps.WebGLReduceLogSum()],
  // ['ReduceMax', '', '1+', () => new reduceOps.WebGLReduceMax()],
  // ['ReduceMean', '', '1+', () => new reduceOps.WebGLReduceMean()],
  // ['ReduceMin', '', '1+', () => new reduceOps.WebGLReduceMin()],
  // ['ReduceProd', '', '1+', () => new reduceOps.WebGLReduceProd()],
  // ['ReduceSum', '', '1+', () => new reduceOps.WebGLReduceSum()],
  // ['ReduceSumSquare', '', '1+', () => new reduceOps.WebGLReduceSumSquare()],
  ['Relu', '', '6+', unaryOps.relu],
  ['Reshape', '', '5+', () => new reshape()],
  // ['Resize', '', '10', () => new WebGLResizePacked(10)],
  // ['Resize', '', '11+', () => new WebGLResizePacked(11)],
  // ['Shape', '', '1+', () => new WebGLShape()],
  ['Sigmoid', '', '6+', unaryOps.sigmoid],
  ['Sin', '', '7+', unaryOps.sin],
  // ['Slice', '', '10+', () => new WebGLSliceV10()],  // TODO: support 'steps' for Slice-10
  // ['Slice', '', '1-9', () => new WebGLSlice()],
  // ['Softmax', '', '1+', () => new WebGLSoftmax()],
  // 'Split' operator has an optional attribute 'split'
  // this attribute determines how the specified axis of input data
  // is split. When the attribute is missing, we need the count of number of outputs
  // so that we can determine the 'split' attribute from the runtime input to the Operator
  // ['Split', '', '2+', (node) => new WebGLSplit(node.outputs.length)],
  ['Sqrt', '', '6+', unaryOps.sqrt],
  // ['Squeeze', '', '1+', () => new WebGLSqueeze()],
  ['Sub', '', '7+', binaryOps.sub],
  // ['Sum', '', '6+', () => new WebGLSum()],  // TODO: support multidirectional broadcast for Sum-8
  ['Tan', '', '7+', unaryOps.tan],
  ['Tanh', '', '6+', unaryOps.tanh],
  // ['Tile', '', '6+', () => new WebGLTile()],
  // ['Transpose', '', '1+', () => new WebGLTranspose()],
  // ['Upsample', '', '7-8', () => new WebGLUpsample(7)],
  // ['Upsample', '', '9', () => new WebGLUpsample(9)],
  // ['Unsqueeze', '', '1+', () => new WebGLUnsqueeze()],
  ['Xor', '', '7+', binaryOps.xor],
];
