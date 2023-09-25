// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';
import {reduceL1Naive, reduceL2Naive, reduceLogSumExpNaive, reduceLogSumNaive, reduceMaxNaive, reduceMeanNaive, reduceMinNaive, reduceProdNaive, reduceSumNaive, reduceSumSquareNaive} from './reduce';
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

const reduceOps: {[key: string]: string} = {
  max: 'select(bestValue, candidate, candidate > bestValue)',
  min: 'select(bestValue, candidate, candidate < bestValue)',
  mean: 'bestValue + candidate',
  sum: 'bestValue + candidate',
  prod: 'bestValue * candidate',
  sumSquare: 'bestValue + candidate * candidate',
  logSumExp: 'bestValue + exp(candidate)',
  l1: 'bestValue + abs(candidate)',
  l2: 'bestValue + candidate * candidate',
  logSum: 'bestValue + candidate'
};

const reduceSharedOps: {[key: string]: string} = {
  max: 'select(bestValue, candidate, candidate > bestValue)',
  min: 'select(bestValue, candidate, candidate < bestValue)',
  mean: 'bestValue + candidate',
  sum: 'bestValue + candidate',
  prod: 'bestValue * candidate',
  sumSquare: 'bestValue + candidate',
  logSumExp: 'bestValue + candidate',
  l1: 'bestValue + candidate',
  l2: 'bestValue + candidate',
  logSum: 'bestValue + candidate'
};

const reduceInitValues: {[key: string]: string} = {
  max: '_A[offset]',
  min: '_A[offset]',
  mean: '0',
  sum: '0',
  prod: '1',
  sumSquare: '0',
  logSumExp: '0',
  l1: '0',
  l2: '0',
  logSum: '0'
};

const reduceOutputValues: {[key: string]: string} = {
  max: 'bestValue',
  min: 'bestValue',
  sum: 'bestValue',
  prod: 'bestValue',
  sumSquare: 'bestValue',
  logSumExp: 'log(bestValue)',
  l1: 'bestValue',
  l2: 'sqrt(bestValue)',
  logSum: 'log(bestValue)'
};

function getInnerMostAxes(numAxes: number, rank: number): number[] {
  const res: number[] = [];
  for (let i = rank - numAxes; i < rank; ++i) {
    res.push(i);
  }
  return res;
}

function computeOutAndReduceShapes(shape: readonly number[], axes: readonly number[]): [number[], number[]] {
  const outputShape = [];
  const rank = shape.length;
  for (let dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      outputShape.push(shape[dim]);
    }
  }
  const reduceShape = axes.map(dim => shape[dim]);
  return [outputShape, reduceShape];
}

function combineLocations(outputLoc: number[], reduceLoc: number[], axes: number[]): number[] {
  const rank = outputLoc.length + reduceLoc.length;
  const loc = [];
  let outIdx = 0;
  let reduceIdx = 0;
    for (let dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      loc.push(outputLoc[outIdx++]);
    } else {
      loc.push(reduceLoc[reduceIdx++]);
    }
  }
  return loc;
}

function expandShapeToKeepDim(shape: number[], axes: number[]): number[] {
  const reduceSubShape = axes.map(x => 1);
  return combineLocations(shape, reduceSubShape, axes);
}

function axesAreInnerMostDims(axes: number[], rank: number): boolean {
  for (let i = 0; i < axes.length; ++i) {
    if (axes[axes.length - i - 1] !== rank - 1 - i) {
      return false;
    }
  }
  return true;
}

function getAxesPermutation(axes: number[], rank: number): number[]|null {
  if (axesAreInnerMostDims(axes, rank)) {
    return null;
  }
  const result: number[] = [];
  for (let i = 0; i < rank; ++i) {
    if (axes.indexOf(i) === -1) {
      result.push(i);
    }
  }
  axes.forEach(axis => result.push(axis));
  return result;
}

function useNaiveReduceMethod(shape: readonly number[], axes: readonly number[], noopWithEmptyAxes: boolean): boolean {
  if (axes.length === 0) {
    return noopWithEmptyAxes ? true : false;
  }

  let outputSize = 1;
  let reduceSize = 1;
  for (let dim = 0; dim < axes.length; dim++) {
    if (axes.indexOf(dim) === -1) {
      outputSize *= shape[dim];
    } else {
      reduceSize *= shape[dim];
    }
  }

  return reduceSize < 32 && outputSize > 1024 ? true : false;
}

export const createReduceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], reduceType: string, outputDataType: DataType,
     outputShape: number[], reduceShape: number[]): ProgramInfo => {
      const inputShape = inputs[0].dims;

      const outputSize = ShapeUtil.size(outputShape);
      const reduceSize = ShapeUtil.size(reduceShape);

      const input = inputVariable('_A', inputs[0].dataType, inputShape);
      const output = outputVariable('output', outputDataType, outputShape);

      const workgroupSize = 32;

      const sharedMemorySnippet = `
         var<workgroup> aBestValues : array<${output.type.storage}, ${workgroupSize}>;
       `;

      const getShaderSource = (shaderHelper: ShaderHelper) => `
        ${shaderHelper.declareVariables(input, output)}
        ${sharedMemorySnippet}
        fn DIV_CEIL(a : u32, b : u32) -> u32 {
          return ((a - 1u) / b + 1u);
         }
        @compute @workgroup_size(${workgroupSize}, 1, 1)
        fn main(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3u) {
          let global_idx = global_id.x;
          let local_idx = local_id.x;

          let outputIndex = global_idx / ${workgroupSize};
          let offset = outputIndex * ${reduceSize};

          var bestValue = ${output.type.storage}(${reduceInitValues[reduceType]});
          let Length = u32(${reduceSize});
          for (var k = local_idx; k < Length && outputIndex < ${outputSize};
             k = k + ${workgroupSize}) {
           let candidate = ${output.type.storage}(${input.getByOffset('offset + k')});
           bestValue = ${reduceOps[reduceType]};
          }
          aBestValues[local_idx] = bestValue;
          workgroupBarrier();

         var reduceSize = min(u32(Length), ${workgroupSize}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (local_idx < currentSize) {
            let candidate = aBestValues[local_idx + interval];
            bestValue = ${reduceSharedOps[reduceType]};
            aBestValues[local_idx] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (local_idx == 0u && outputIndex < ${outputSize}) {
          ${
          output.setByOffset(
              'outputIndex',
              `${
                  reduceType === 'mean' ? `bestValue / ${output.type.storage}(${reduceSize})` :
                                          `${reduceOutputValues[reduceType]}`}`)};
         }
        }`;

      return {
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType: outputDataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize)})
      };
    };

const createReduceAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ReduceAttributes): ReduceAttributes => {
      const axes: number[] = [];
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => axes.push(Number(v)));
      }
      return createAttributeWithCacheKey(
          {axes, keepDims: attributes.keepDims, noopWithEmptyAxes: attributes.noopWithEmptyAxes});
    };

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ReduceAttributes, reduceType: string,
     resOutShape: number[], reduceShape: number[]): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {
        name,
        inputTypes: [GpuDataType.default],
        cacheHint: attributes.cacheKey + '_' + inputs[0].dims.map(d => d.toString()).join(',')
      };
      return {
        ...metadata,
        get: () =>
            createReduceProgramInfo(metadata, [inputs[0]], reduceType, inputs[0].dataType, resOutShape, reduceShape)
      };
    };

const reduceCommon =
    (context: ComputeContext, name: string, attributes: ReduceAttributes,
     reduceType: 'sum'|'sumSquare'|'prod'|'min'|'max'|'mean'|'logSumExp'|'l1'|'l2'|'logSum'): void => {
      const updatedAttributes: ReduceAttributes =
          context.inputs.length === 1 ? attributes : createReduceAttributesFromInputs(context.inputs, attributes);

      let updatedAxes = updatedAttributes.axes;
      if (updatedAxes.length === 0 && !updatedAttributes.noopWithEmptyAxes) {
        updatedAxes = context.inputs[0].dims.map((s, i) => i);
      }
      const normalizeAxes = ShapeUtil.normalizeAxes(updatedAxes, context.inputs[0].dims.length);

      let axes = normalizeAxes;
      let input = context.inputs[0];
      const permutedAxes = getAxesPermutation(axes, context.inputs[0].dims.length);
      if (permutedAxes != null) {
        const inputTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: permutedAxes});
        input = context.compute(
            {
              ...transposeProgramMetadata,
              cacheHint: inputTransposeAttribute.cacheKey,
              get: () => createTransposeProgramInfo(context.inputs[0], inputTransposeAttribute.perm)
            },
            {inputs: [0], outputs: [-2]})[0];
        axes = getInnerMostAxes(axes.length, input.dims.length);
      }

      const [outputShape, reduceShape] = computeOutAndReduceShapes(input.dims, axes);
      let finalOutputShape = outputShape;
      if (updatedAttributes.keepDims) {
        finalOutputShape = expandShapeToKeepDim(outputShape, normalizeAxes);
      }

      context.compute(
          createReduceProgramInfoLoader([input], name, updatedAttributes, reduceType, finalOutputShape, reduceShape),
          {inputs: [input]});
    };

export const reduceMean = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceMeanNaive(context, attributes) :
      reduceCommon(context, 'ReduceMean', attributes, 'mean');
};

export const reduceL1 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceL1Naive(context, attributes) :
      reduceCommon(context, 'ReduceL1', attributes, 'l1');
};

export const reduceL2 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceL2Naive(context, attributes) :
      reduceCommon(context, 'ReduceL2', attributes, 'l2');
};

export const reduceLogSumExp = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceLogSumExpNaive(context, attributes) :
      reduceCommon(context, 'ReduceLogSumExp', attributes, 'logSumExp');
};

export const reduceMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceMaxNaive(context, attributes) :
      reduceCommon(context, 'ReduceMax', attributes, 'max');
};

export const reduceMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceMinNaive(context, attributes) :
      reduceCommon(context, 'ReduceMin', attributes, 'min');
};

export const reduceProd = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceProdNaive(context, attributes) :
      reduceCommon(context, 'ReduceProd', attributes, 'prod');
};

export const reduceSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceSumNaive(context, attributes) :
      reduceCommon(context, 'ReduceSum', attributes, 'sum');
};

export const reduceSumSquare = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceSumSquareNaive(context, attributes) :
      reduceCommon(context, 'ReduceSumSquare', attributes, 'sumSquare');
};

export const reduceLogSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  useNaiveReduceMethod(context.inputs[0].dims, attributes.axes, attributes.noopWithEmptyAxes) ?
      reduceLogSumNaive(context, attributes) :
      reduceCommon(context, 'ReduceLogSum', attributes, 'logSum');
};

export const parseReduceAttributes = (attributes: Record<string, unknown>): ReduceAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ReduceAttributes, keyof AttributeWithCacheKey>);
