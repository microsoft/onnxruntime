// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {FLOAT_TYPES, NUMBER_TYPES} from '../../operators';
import {OpSet} from '../../opset';

import {WebGLBatchNormalization} from './ops/batch-normalization';
import * as binaryOps from './ops/binary-op';
import {WebGLClip} from './ops/clip';
import {WebGLConcat} from './ops/concat';
import {WebGLConv} from './ops/conv';
import {WebGLDepthToSpace} from './ops/depth-to-space';
import {WebGLDropout} from './ops/dropout';
import {WebGLElu} from './ops/elu';
import {WebGLFlatten} from './ops/flatten';
import {WebGLGather} from './ops/gather';
import {WebGLGemm} from './ops/gemm';
import {WebGLImageScaler} from './ops/image-scaler';
import {WebGLInstanceNormalization} from './ops/instance-normalization';
import {WebGLLeakyRelu} from './ops/leaky-relu';
import {WebGLMatMul} from './ops/matmul';
import {WebGLPad} from './ops/pad';
import {WebGLAveragePool, WebGLGlobalAveragePool, WebGLGlobalMaxPool, WebGLMaxPool} from './ops/pool';
import * as reduceOps from './ops/reduce';
import {WebGLReshape} from './ops/reshape';
import {WebGLResizePacked} from './ops/resize-packed';
import {WebGLShape} from './ops/shape';
import {WebGLSlice, WebGLSliceV10} from './ops/slice';
import {WebGLSoftmax} from './ops/softmax';
import {WebGLSplit} from './ops/split';
import {WebGLSqueeze} from './ops/squeeze';
import {WebGLSum} from './ops/sum';
import {WebGLTile} from './ops/tile';
import {WebGLTranspose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {WebGLUnsqueeze} from './ops/unsqueeze';
import {WebGLUpsample} from './ops/upsample';

export const WEBGL_OP_RESOLVE_RULES: readonly OpSet.ResolveRule[] = [
  ['Abs', '', '6+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslAbs())],
  ['Acos', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAcos())],
  ['Add', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslAdd())],
  ['And', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslAnd())],
  ['Asin', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAsin())],
  ['Atan', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAtan())],
  ['AveragePool', '', '7-10', () => new WebGLAveragePool()],  // TODO: support new attributes for AveragePool-10
  ['BatchNormalization', '', '7+', () => new WebGLBatchNormalization()],
  ['Ceil', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCeil())],
  ['Clip', '', '6-10', () => new WebGLClip()],
  ['Concat', '', '4+', () => new WebGLConcat()],
  ['Conv', '', '1+', () => new WebGLConv()],
  ['Cos', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCos())],
  ['Div', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslDiv())],
  ['Dropout', '', '7+', () => new WebGLDropout()],
  ['DepthToSpace', '', '1+', () => new WebGLDepthToSpace()],
  ['Equal', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslEqual(), undefined, 'bool')],
  ['Elu', '', '6+', () => new WebGLElu()],
  ['Exp', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslExp())],
  ['Flatten', '', '1+', () => new WebGLFlatten()],
  ['Floor', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslFloor())],
  ['Gather', '', '1+', () => new WebGLGather()],
  ['Gemm', '', '7-10', () => new WebGLGemm(false)],
  ['Gemm', '', '11+', () => new WebGLGemm(true)],
  ['GlobalAveragePool', '', '1+', () => new WebGLGlobalAveragePool()],
  ['GlobalMaxPool', '', '1+', () => new WebGLGlobalMaxPool()],
  ['Greater', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslGreater(), undefined, 'bool')],
  ['Identity', '', '1+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslIdentity())],
  ['ImageScaler', '', '1+', () => new WebGLImageScaler()],
  ['InstanceNormalization', '', '6+', () => new WebGLInstanceNormalization()],
  ['LeakyRelu', '', '6+', () => new WebGLLeakyRelu()],
  ['Less', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslLess(), undefined, 'bool')],
  ['Log', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslLog())],
  ['MatMul', '', '1+', () => new WebGLMatMul()],
  ['MaxPool', '', '1-9', () => new WebGLMaxPool()],  // TODO: support new attributes for MaxPool-8 and MaxPool-10
  ['Mul', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslMul())],
  ['Neg', '', '6+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslNeg())],
  ['Not', '', '1+', () => new unaryOps.WebGLUnaryOp(['bool'], unaryOps.glslNot())],
  ['Or', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslOr())],
  ['Pad', '', '2-10', () => new WebGLPad()],
  ['Pow', '', '7+', () => new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPow())],
  ['PRelu', '', '7+', () => new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPRelu())],
  ['ReduceLogSum', '', '1+', () => new reduceOps.WebGLReduceLogSum()],
  ['ReduceMax', '', '1+', () => new reduceOps.WebGLReduceMax()],
  ['ReduceMean', '', '1+', () => new reduceOps.WebGLReduceMean()],
  ['ReduceMin', '', '1+', () => new reduceOps.WebGLReduceMin()],
  ['ReduceProd', '', '1+', () => new reduceOps.WebGLReduceProd()],
  ['ReduceSum', '', '1+', () => new reduceOps.WebGLReduceSum()],
  ['ReduceSumSquare', '', '1+', () => new reduceOps.WebGLReduceSumSquare()],
  ['Relu', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslRelu())],
  ['Reshape', '', '5+', () => new WebGLReshape()],
  ['Resize', '', '10', () => new WebGLResizePacked(10)],
  ['Resize', '', '11+', () => new WebGLResizePacked(11)],
  ['Shape', '', '1+', () => new WebGLShape()],
  ['Sigmoid', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSigmoid())],
  ['Sin', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSin())],
  ['Slice', '', '10+', () => new WebGLSliceV10()],  // TODO: support 'steps' for Slice-10
  ['Slice', '', '1-9', () => new WebGLSlice()],
  ['Softmax', '', '1+', () => new WebGLSoftmax()],
  // 'Split' operator has an optional attribute 'split'
  // this attribute determines how the specified axis of input data
  // is split. When the attribute is missing, we need the count of number of outputs
  // so that we can determine the 'split' attribute from the runtime input to the Operator
  ['Split', '', '2+', (node) => new WebGLSplit(node.outputs.length)],
  ['Sqrt', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSqrt())],
  ['Squeeze', '', '1+', () => new WebGLSqueeze()],
  ['Sub', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslSub())],
  ['Sum', '', '6+', () => new WebGLSum()],  // TODO: support multidirectional broadcast for Sum-8
  ['Tan', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslTan())],
  ['Tanh', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslTanh())],
  ['Tile', '', '6+', () => new WebGLTile()],
  ['Transpose', '', '1+', () => new WebGLTranspose()],
  ['Upsample', '', '7-8', () => new WebGLUpsample(7)],
  ['Upsample', '', '9', () => new WebGLUpsample(9)],
  ['Unsqueeze', '', '1+', () => new WebGLUnsqueeze()],
  ['Xor', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslXor())],
];
