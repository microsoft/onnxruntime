// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, TensorInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper, WORKGROUP_SIZE} from './common';

export interface BatchNormAttributes extends AttributeWithCacheKey {
  readonly epsilon: number;
  readonly momentum: number;
  readonly spatial: boolean;
  readonly trainingMode: boolean;
  readonly format: 'nhwc'|'nchw';
  readonly outputCount: number;
}

const validateInputs = (inputs: readonly TensorView[], attributes: BatchNormAttributes): void => {
  if (!inputs || inputs.length !== 5) {
    throw new Error('BatchNormalization requires 5 inputs');
  }

  const checkShapeEqual = (actual: readonly number[], expected: readonly number[], message: string) => {
    const r = expected.length;
    if (r !== actual.length) {
      throw new Error(`${message}: num dimensions != ${r}`);
    }
    expected.forEach((v, i) => {
      if (v !== actual[i]) {
        switch (i % 100) {
          case 11:
          case 12:
          case 13:
            throw new Error(`${message}: ${i}th dimension != ${v}`);
          default:
        }
        switch (i % 10) {
          case 1:
            throw new Error(`${message}: ${i}st dimension != ${v}`);
          case 2:
            throw new Error(`${message}: ${i}nd dimension != ${v}`);
          case 3:
            throw new Error(`${message}: ${i}rd dimension != ${v}`);
          default:
            throw new Error(`${message}: ${i}th dimension != ${v}`);
        }
      }
    });
  };

  if (inputs[0].dims.length > 1) {
    const shape = Object.seal<Record<typeof attributes.format, number[]>>({
      nhwc: inputs[0].dims.slice(-1),
      nchw: inputs[0].dims.slice(1, attributes.spatial ? 2 : undefined),
    })[attributes.format];
    checkShapeEqual(inputs[1].dims, shape, 'Invalid input scale');
    checkShapeEqual(inputs[2].dims, shape, 'Invalid input B');
    checkShapeEqual(inputs[3].dims, shape, 'Invalid input mean');
    checkShapeEqual(inputs[4].dims, shape, 'Invalid input var');
  } else {
    checkShapeEqual(inputs[1].dims, [1], 'Invalid input scale');
    checkShapeEqual(inputs[2].dims, [1], 'Invalid input B');
    checkShapeEqual(inputs[3].dims, [1], 'Invalid input mean');
    checkShapeEqual(inputs[4].dims, [1], 'Invalid input var');
  }
};

const createBatchNormProgramInfo = (inputs: readonly TensorView[], attributes: BatchNormAttributes): ProgramInfo => {
  const {epsilon, momentum, spatial, trainingMode, format, outputCount} = attributes;
  const shape = inputs[0].dims;

  const outputs = Object.seal<Record<typeof format, TensorInfo[]>>({
    nhwc: [
      [inputs[0].dataType, shape, GpuDataType.default],
      [(inputs[3] ?? inputs[0]).dataType, shape.length > 1 ? shape.slice(-1) : [1], GpuDataType.default],
      [(inputs[4] ?? inputs[0]).dataType, shape.length > 1 ? shape.slice(-1) : [1], GpuDataType.default],
      [(inputs[3] ?? inputs[0]).dataType, shape.length > 1 ? shape.slice(-1) : [1], GpuDataType.default],
      [(inputs[4] ?? inputs[0]).dataType, shape.length > 1 ? shape.slice(-1) : [1], GpuDataType.default],
    ].map(([dataType, dims, gpuDataType]: [number, number[], GpuDataType]) => ({dataType, dims, gpuDataType})),

    nchw: [
      [
        inputs[0].dataType,
        shape,
        GpuDataType.default,
      ],
      [
        (inputs[3] ?? inputs[0]).dataType,
        shape.length > 1 ? shape.slice(1, spatial ? 2 : undefined) : [1],
        GpuDataType.default,
      ],
      [
        (inputs[4] ?? inputs[0]).dataType,
        shape.length > 1 ? shape.slice(1, spatial ? 2 : undefined) : [1],
        GpuDataType.default,
      ],
      [
        (inputs[3] ?? inputs[0]).dataType,
        shape.length > 1 ? shape.slice(1, spatial ? 2 : undefined) : [1],
        GpuDataType.default,
      ],
      [
        (inputs[4] ?? inputs[0]).dataType,
        shape.length > 1 ? shape.slice(1, spatial ? 2 : undefined) : [1],
        GpuDataType.default,
      ],
    ].map(([dataType, dims, gpuDataType]: [number, number[], GpuDataType]) => ({dataType, dims, gpuDataType})),
  })[format];

  const xShape = Object.seal<Record<typeof format, number[]>>({
    nhwc: [
      ShapeUtil.size(shape.slice(0, Math.max(1, shape.length - 1))),
      ShapeUtil.size(shape.slice(Math.max(1, shape.length - 1))),
      1,
    ],

    nchw: spatial ?

        [
          ShapeUtil.size(shape.slice(0, 1)),
          ShapeUtil.size(shape.slice(1, 2)),
          ShapeUtil.size(shape.slice(2)),
        ] :
        [
          ShapeUtil.size(shape.slice(0, 1)),
          ShapeUtil.size(shape.slice(1)),
          1,
        ],
  })[format];
  const x = inputVariable('x', inputs[0].dataType, xShape);
  const scale = inputVariable('scale', inputs[1].dataType, [ShapeUtil.size(inputs[1].dims)]);
  const bias = inputVariable('bias', inputs[2].dataType, [ShapeUtil.size(inputs[2].dims)]);
  const inputMean = outputCount <= 3 ?
      inputVariable('inputMean', inputs[3].dataType, [ShapeUtil.size(inputs[3].dims)]) :
      inputVariable('_inputMean', inputs[0].dataType, [0]);
  const inputVar = outputCount <= 3 ? inputVariable('inputVar', inputs[4].dataType, [ShapeUtil.size(inputs[4].dims)]) :
                                      inputVariable('_inputVar', inputs[0].dataType, [0]);
  const y = outputVariable('y', outputs[0].dataType, xShape);
  const runningMean = outputVariable('runningMean', outputs[1].dataType, [ShapeUtil.size(outputs[1].dims)]);
  const runningVar = outputVariable('runningVar', outputs[2].dataType, [ShapeUtil.size(outputs[2].dims)]);
  const savedMean = outputCount <= 3 ?
      outputVariable('_savedMean', outputs[3].dataType, [0]) :
      outputVariable('savedMean', outputs[3].dataType, [ShapeUtil.size(outputs[3].dims)]);
  const savedVar = outputCount <= 3 ?
      outputVariable('_savedVar', outputs[4].dataType, [0]) :
      outputVariable('savedVar', outputs[4].dataType, [ShapeUtil.size(outputs[4].dims)]);
  const workgroupSizeAlongAxis = (axis: number) => xShape.indexOf(Math.max(...xShape)) === axis ? WORKGROUP_SIZE : 1;
  const workgroupSize = [workgroupSizeAlongAxis(0), workgroupSizeAlongAxis(1), workgroupSizeAlongAxis(2)] as const;
  const inputVariables = inputs.length > 3 ? [x, scale, bias, inputMean, inputVar] : [x, scale, bias];
  const outputVariables =
      outputCount > 3 ? [y, runningMean, runningVar, savedMean, savedVar] : [y, runningMean, runningVar];
  const getInferenceModeShaderSource = (helper: ShaderHelper) => `
      alias T = ${x.type.value};

      const epsilon = ${epsilon};
      const inputShape = ${x.indices(...xShape)};
      const outputShape = ${y.indices(...xShape)};
      const workgroupSize = vec3<u32>(${workgroupSize});

      const local_id = vec3(0);
      const local_index = 0;
      const global_id = vec3(0);

      var<workgroup> workgroupIdOutOfBounds: bool;

      struct SharedMem {
        scale: array<T, workgroupSize.y>,
        bias: array<T, workgroupSize.y>,
      }

      var<workgroup> sharedMem: SharedMem;

      ${helper.declareVariables(...inputVariables, y)}

      ${helper.mainStart([...workgroupSize])}
        let is1DimensionDispatch =
            ((local_id.x + 0xffffffff == local_id.x - 1) && (local_index + 0xffffffff != local_index - 1) &&
             (global_id.x + 0xffffffff == global_id.x - 1));
        if (is1DimensionDispatch ==
            ((local_id.x + 0xffffffff != local_id.x - 1) && (local_index + 0xffffffff == local_index - 1) &&
             (global_id.x + 0xffffffff != global_id.x - 1))) {
          while (${y.getByOffset(0)} == ${y.getByOffset(0)}) {
          }
        }
        let numWorkgroups = (inputShape - 1) / workgroupSize + 1;
        let workgroupIndex = global_idx /
            select(workgroupSize.x * workgroupSize.y * workgroupSize.z, workgroupSize.x, is1DimensionDispatch);
        let workgroupId =
            workgroupIndex / vec3<u32>(1, numWorkgroups.x, numWorkgroups.x * numWorkgroups.y) % numWorkgroups;
        let localId =
            local_id | (local_index / vec3<u32>(1, workgroupSize.x, workgroupSize.x * workgroupSize.y) % workgroupSize);
        let localIndex = local_index | dot(local_id, vec3<u32>(1, workgroupSize.x, workgroupSize.x * workgroupSize.y));
        let globalId = workgroupId * workgroupSize + localId;
        if (localIndex == 0) {
          workgroupIdOutOfBounds = workgroupIndex >= numWorkgroups.x * numWorkgroups.y * numWorkgroups.z;
        }
        if (workgroupUniformLoad(&workgroupIdOutOfBounds)) {
          return;
        }

        let i = globalId.y - localId.y + localIndex;
        if (localIndex < workgroupSize.y && i < inputShape.y) {
          let scale = ${scale.getByIndices('i')};
          let bias = ${bias.getByIndices('i')};
          let inputMean = ${inputMean.getByIndices('i')};
          let inputVar = ${inputVar.getByIndices('i')};
          sharedMem.scale[localIndex] = scale * inverseSqrt(inputVar + epsilon);
          sharedMem.bias[localIndex] = fma(-inputMean, scale * inverseSqrt(inputVar + epsilon), bias);
        }
        workgroupBarrier();

        let scale = sharedMem.scale[localId.y];
        let bias = sharedMem.bias[localId.y];
        if (all(globalId < inputShape)) {
          let x = ${x.getByIndices('globalId')};
          ${y.getByOffset(y.indicesToOffset('globalId'))} = fma(x, scale, bias);
        }
      }`;

  const getTrainingModeShaderSource = (helper: ShaderHelper) => `
      alias T = ${x.type.value};

      const epsilon = ${epsilon};
      const momentum = ${momentum};
      const inputShape = ${x.indices(...xShape)};
      const outputShape = ${y.indices(...xShape)};
      const workgroupSize = vec3<u32>(${workgroupSize});

      const local_id = vec3(0);
      const local_index = 0;
      const global_id = vec3(0);

      var<workgroup> workgroupIdOutOfBounds: bool;

      struct SharedMem {
        scale: array<T, workgroupSize.y>,
        bias: array<T, workgroupSize.y>,
        sum: array<array<array<T, workgroupSize.x>, workgroupSize.y>, workgroupSize.z>,
        squareSum: array<array<array<T, workgroupSize.x>, workgroupSize.y>, workgroupSize.z>,
      }

      var<workgroup> sharedMem: SharedMem;

      ${helper.declareVariables(...inputVariables, ...outputVariables)}

      fn prevPowOf2(n: u32) -> u32 {
        return u32(1 << (32 - countLeadingZeros(n >> 1)));
      }

      ${helper.mainStart([...workgroupSize])}
        let is1DimensionDispatch =
            ((local_id.x + 0xffffffff == local_id.x - 1) && (local_index + 0xffffffff != local_index - 1) &&
             (global_id.x + 0xffffffff == global_id.x - 1));
        if (is1DimensionDispatch ==
            ((local_id.x + 0xffffffff != local_id.x - 1) && (local_index + 0xffffffff == local_index - 1) &&
             (global_id.x + 0xffffffff != global_id.x - 1))) {
          while (${y.getByOffset(0)} == ${y.getByOffset(0)}) {
          }
        }
        let numWorkgroups = (inputShape.y - 1) / workgroupSize.y + 1;
        let workgroupIndex = global_idx /
            select(workgroupSize.x * workgroupSize.y * workgroupSize.z, workgroupSize.x, is1DimensionDispatch);
        let workgroupId = vec3<u32>(0, workgroupIndex, 0);
        let localId =
            local_id | (local_index / vec3<u32>(1, workgroupSize.x, workgroupSize.x * workgroupSize.y) % workgroupSize);
        let localIndex = local_index | dot(local_id, vec3<u32>(1, workgroupSize.x, workgroupSize.x * workgroupSize.y));
        let globalId = workgroupId * workgroupSize + localId;
        if (localIndex == 0) {
          workgroupIdOutOfBounds = workgroupIndex >= numWorkgroups;
        }
        if (workgroupUniformLoad(&workgroupIdOutOfBounds)) {
          return;
        }

        var sum = T(0);
        var squareSum = T(0);
        if (globalId.y < inputShape.y) {
          for (var k = localId.z; k < inputShape.z; k += workgroupSize.z) {
            for (var i = localId.x; i < inputShape.x; i += workgroupSize.x) {
              let x = ${x.getByIndices(x.indices('i', 'globalId.y', 'k'))};
              sum += x;
              squareSum += x * x;
            }
          }
        }
        sharedMem.sum[localId.z][localId.y][localId.x] = sum;
        sharedMem.squareSum[localId.z][localId.y][localId.x] = squareSum;
        workgroupBarrier();

        if (localId.x < workgroupSize.x - prevPowOf2(workgroupSize.x - 1)) {
          sharedMem.sum[localId.z][localId.y][localId.x] +=
              sharedMem.sum[localId.z][localId.y][workgroupSize.x - 1 - localId.x];
          sharedMem.squareSum[localId.z][localId.y][localId.x] +=
              sharedMem.squareSum[localId.z][localId.y][workgroupSize.x - 1 - localId.x];
        }
        workgroupBarrier();
        for (var size = prevPowOf2(workgroupSize.x - 1) >> 1; size > 0; size >>= 1) {
          if (localId.x < size) {
            sharedMem.sum[localId.z][localId.y][localId.x] += sharedMem.sum[localId.z][localId.y][localId.x + size];
            sharedMem.squareSum[localId.z][localId.y][localId.x] +=
                sharedMem.squareSum[localId.z][localId.y][localId.x + size];
          }
          workgroupBarrier();
        }
        if (localId.z < workgroupSize.z - prevPowOf2(workgroupSize.z - 1)) {
          sharedMem.sum[localId.z][localId.y][localId.x] +=
              sharedMem.sum[workgroupSize.z - 1 - localId.z][localId.y][localId.x];
          sharedMem.squareSum[localId.z][localId.y][localId.x] +=
              sharedMem.squareSum[workgroupSize.z - 1 - localId.z][localId.y][localId.x];
        }
        workgroupBarrier();
        for (var size = prevPowOf2(workgroupSize.z - 1) >> 1; size > 0; size >>= 1) {
          if (localId.z < size) {
            sharedMem.sum[localId.z][localId.y][localId.x] += sharedMem.sum[localId.z + size][localId.y][localId.x];
            sharedMem.squareSum[localId.z][localId.y][localId.x] +=
                sharedMem.squareSum[localId.z + size][localId.y][localId.x];
          }
          workgroupBarrier();
        }

        let i = globalId.y - localId.y + localIndex;
        if (localIndex < workgroupSize.y && i < inputShape.y) {
          let scale = ${scale.getByIndices('i')};
          let bias = ${bias.getByIndices('i')};
          let currentMean = sharedMem.sum[0][localIndex][0] / T(inputShape.x * inputShape.z);
          let currentVar =
              sharedMem.squareSum[0][localIndex][0] / T(inputShape.x * inputShape.z) - currentMean * currentMean;
          sharedMem.scale[localIndex] = scale * inverseSqrt(currentVar + epsilon);
          sharedMem.bias[localIndex] = fma(-currentMean, scale * inverseSqrt(currentVar + epsilon), bias);
          ${
      outputCount > 3 ? `
          ${runningMean.getByOffset(runningMean.indicesToOffset('i'))} = mix(currentMean, 0, momentum);
          ${runningVar.getByOffset(runningVar.indicesToOffset('i'))} = mix(currentVar, 0, momentum);
          ${savedMean.getByOffset(savedMean.indicesToOffset('i'))} = currentMean;
          ${savedVar.getByOffset(savedVar.indicesToOffset('i'))} = inverseSqrt(currentVar + epsilon);
          ` :
                        `
          let inputMean = ${inputMean.getByIndices('i')};
          let inputVar = ${inputVar.getByIndices('i')};
          ${runningMean.getByOffset(runningMean.indicesToOffset('i'))} = mix(currentMean, inputMean, momentum);
          ${runningVar.getByOffset(runningVar.indicesToOffset('i'))} = mix(currentVar, inputVar, momentum);
          `}
        }
        workgroupBarrier();

        let scale = sharedMem.scale[localId.y];
        let bias = sharedMem.bias[localId.y];
        if (globalId.y < inputShape.y) {
          for (var k = localId.z; k < inputShape.z; k += workgroupSize.z) {
            for (var i = localId.x; i < inputShape.x; i += workgroupSize.x) {
              let x = ${x.getByIndices(x.indices('i', 'globalId.y', 'k'))};
              ${y.getByOffset(y.indicesToOffset(y.indices('i', 'globalId.y', 'k')))} = fma(x, scale, bias);
            }
          }
        }
      }`;

  if (trainingMode) {
    return {
      name: 'BatchNormalization',
      shaderCache: {hint: attributes.cacheKey},
      getShaderSource: getTrainingModeShaderSource,
      getRunData: () => ({
        outputs: outputs.slice(0, outputCount),
        dispatchGroup: {
          x: 1,
          y: Math.ceil(xShape[1] / workgroupSize[1]),
          z: 1,
        },
      }),
    };
  } else {
    return {
      name: 'BatchNormalization',
      shaderCache: {hint: attributes.cacheKey},
      getShaderSource: getInferenceModeShaderSource,
      getRunData: () => ({
        outputs: outputs.slice(0, outputCount),
        dispatchGroup: {
          x: Math.ceil(xShape[0] / workgroupSize[0]),
          y: Math.ceil(xShape[1] / workgroupSize[1]),
          z: Math.ceil(xShape[2] / workgroupSize[2]),
        },
      }),
    };
  }
};

const createPostBatchNormProgramInfo =
    (inputs: readonly TensorView[], attributes: BatchNormAttributes): ProgramInfo => {
      const {momentum} = attributes;

      const inputShape = [Math.ceil(ShapeUtil.size(inputs[0].dims) / 4)];
      const input = inputVariable('input', inputs[0].dataType, inputShape, 4);
      const running = outputVariable('running', inputs[1].dataType, [Math.ceil(ShapeUtil.size(inputs[1].dims) / 4)], 4);

      const getShaderSource = (helper: ShaderHelper) => `
      const momentum = ${momentum};

      ${helper.declareVariables(input, running)}

      ${helper.mainStart()}
        let size = arrayLength(&${input.name});
        ${helper.guardAgainstOutOfBoundsWorkgroupSizes('size')}
        ${running.getByOffset(running.indicesToOffset('global_idx'))} += ${
          input.getByIndices('global_idx')} * momentum;
      }`;

      return {
        name: 'PostBatchNormalization',
        getShaderSource,
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [],
          dispatchGroup: {x: Math.ceil(inputShape[0] / WORKGROUP_SIZE)},
        }),
      };
    };

export const parseBatchNormAttributes = (attributes: Record<string, unknown>): BatchNormAttributes =>
    createAttributeWithCacheKey(attributes as Omit<BatchNormAttributes, keyof AttributeWithCacheKey>);

export const batchNorm = (context: ComputeContext, attributes: Record<string, unknown>): void => {
  const {inputs, outputCount} = context;
  const updatedAttributes = parseBatchNormAttributes({...attributes, outputCount});
  validateInputs(inputs, updatedAttributes);
  if (outputCount <= 3) {
    context.compute(createBatchNormProgramInfo(inputs, updatedAttributes));
  } else {
    const [x, scale, bias, inputMean, inputVar] = inputs;
    const [, runningMean, runningVar] =
        context.compute(createBatchNormProgramInfo([x, scale, bias], updatedAttributes), {
          inputs: [x, scale, bias],
        });
    context.compute(createPostBatchNormProgramInfo([inputMean, runningMean], updatedAttributes), {
      inputs: [inputMean, runningMean],
    });
    context.compute(createPostBatchNormProgramInfo([inputVar, runningVar], updatedAttributes), {
      inputs: [inputVar, runningVar],
    });
  }
};
