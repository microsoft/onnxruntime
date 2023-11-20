// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

type CoordinateTransformMode = 'half_pixel'|'asymmetric'|'pytorch_half_pixel'|'tf_half_pixel_for_nn'|'align_corners'|
    'tf_crop_and_resize'|'half_pixel_symmetric';

type KeepAspectRatioPolicy = 'stretch'|'not_smaller'|'not_larger';

type Mode = 'nearest'|'linear'|'cubic';

type NearestMode = 'round_prefer_floor'|'round_prefer_ceil'|'floor'|'ceil'|'simple';

export interface ResizeAttributes extends AttributeWithCacheKey {
  antialias: number;
  axes: number[];
  coordinateTransformMode: CoordinateTransformMode;
  cubicCoeffA: number;
  excludeOutside: boolean;
  extrapolationValue: number;
  keepAspectRatioPolicy: KeepAspectRatioPolicy;
  mode: Mode;
  nearestMode: NearestMode;
}

const validateScales = (scales: number[], attributes: ResizeAttributes): void => {
  scales.every((value) => value > 0 || (() => {
                            throw new Error('Resize requires scales input values to be positive');
                          }));
  // Check scales dims based on mode: LINEAR, CUBIC
  if (scales.length > 0) {
    if (attributes.mode === 'linear') {
      if (!(scales.length === 2 || (scales.length === 4 && scales[0] === 1 && scales[1] === 1) ||
            (scales.length === 4 && scales[0] === 1 && scales[3] === 1))) {
        throw new Error('Resize requires scales input size to be 2 or 4 for linear mode');
      }
    } else if (attributes.mode === 'cubic') {
      if (!(scales.length === 2 || (scales.length === 4 && scales[0] === 1 && scales[1] === 1) ||
            (scales.length === 4 && scales[0] === 1 && scales[3] === 1))) {
        throw new Error('Resize requires scales input size to be 2 or 4 for cubic mode');
      }
    }
  }
};

const updateScales = (scales: readonly number[], axes: readonly number[], rank: number): number[] => {
  axes.every((value) => value >= 0 && value < rank || (() => {
                          throw new Error('Resize requires axes input values to be positive and less than rank');
                        }));
  const newScales = new Array(rank).fill(1.0);
  axes.forEach((value, index) => newScales[value] = scales[index]);
  return newScales;
};

const validateInputs =
    (inputs: readonly TensorView[], attributes: ResizeAttributes, opsetVersion: number, scales: number[],
     sizes: number[], roi: number[]): void => {
      const [roiInputIndex, scalesInputIndex, sizesInputIndex] =
          (opsetVersion > 10) ? [1, 2, 3] : [-1, (inputs.length > 1) ? 1 : -1, -1];
      const rank = inputs[0].dims.length;
      if (roiInputIndex > 0 && inputs.length > roiInputIndex && inputs[roiInputIndex].dims.length > 0) {
        inputs[roiInputIndex].getFloat32Array().forEach((value) => roi.push(value));

      } else if (attributes.coordinateTransformMode === 'tf_crop_and_resize') {
        throw new Error('Resize requires RoI input to be specified when coordinateTransformMode is tfCropAndResize');
      }

      if (scalesInputIndex > 0 && inputs.length > scalesInputIndex && inputs[scalesInputIndex].dims.length > 0) {
        inputs[scalesInputIndex].getFloat32Array().forEach((value) => scales.push(value));
        if (scales.length !== 0 &&
            (scales.length !== rank && (opsetVersion >= 18 && scales.length !== attributes.axes.length))) {
          throw new Error(
              'Resize requires scales input size to be same as input rank or axes size for opset 18 and up');
        }
        validateScales(scales, attributes);
        if (attributes.axes.length > 0) {
          updateScales(scales, attributes.axes, rank).forEach((value, index) => scales[index] = value);
        }
      }
      if (sizesInputIndex > 0 && inputs.length > sizesInputIndex) {
        inputs[sizesInputIndex].getBigInt64Array().forEach((value) => sizes.push(Number(value)));
        if (sizes.length !== rank || (opsetVersion >= 18 && sizes.length === attributes.axes.length)) {
          throw new Error('Resize requires sizes input size to be same as input rank or axes size for opset 18 and up');
        }
      }

      if (attributes.axes.length > 0) {
        if (scales.length !== attributes.axes.length) {
          throw new Error('Resize requires "scales" input size to be of axes rank when axes attributes is specified');
        }
        if (sizes.length !== attributes.axes.length) {
          throw new Error(
              'Resize requires "sizes" input size to be of rank axes rank when axes attributes is specified');
        }
      }
      if (typeof scales !== 'undefined' && typeof sizes !== 'undefined' && scales.length > 0 && sizes.length > rank) {
        throw new Error('Resize requires only of scales or sizes to be specified');
      }
    };

const getOriginalCoordinateFromResizedCoordinate = (coordinateTransferMode: CoordinateTransformMode): string =>
    'fn getOriginalCoordinateFromResizedCoordinate(xResized: f32, xScale: f32, lengthResized: f32,\
    lengthOriginal: f32, roiStart: f32, roiEnd: f32) -> f32 { ' +
    (() => {
      switch (coordinateTransferMode) {
        case 'asymmetric':
          return 'return xResized / xScale;';
        case 'pytorch_half_pixel':
          return 'if (lengthResized > 1) { \
                    return (xResized + 0.5) / xScale - 0.5; \
                  } else { \
                    return 0.0; \
                  }';
        case 'tf_half_pixel_for_nn':
          return 'return (xResized + 0.5) / xScale;';
        case 'align_corners':
          return 'if (lengthResized == 1) { \
                    return 0.0; \
                  } else { \
                    return xResized * (lengthOriginal - 1) / (lengthResized - 1); \
                  }';
        case 'tf_crop_and_resize':
          return 'if (lengthResized > 1) { \
                    return roiStart * (lengthOriginal - 1) + \
                          (xResized * (roiEnd - roiStart) * (lengthOriginal - 1)) / (lengthResized - 1); \
                  } else { \
                    return 0.5 * (roiStart + roiEnd) * f32(lengthOriginal - 1); \
                  }';
        case 'half_pixel_symmetric':
          return [
            'const outputWidth = xScale * lengthResized;', 'const adjustment = lengthResized / outputWidth;',
            'const center = lengthOriginal / 2;', 'const offset = center * (1 - adjustment);',
            'return offset + ((xResized + 0.5) / xScale) - 0.5;'
          ].join('\n');
        case 'half_pixel':
          return 'return ((xResized + 0.5) / xScale) - 0.5;';
        default:
          throw new Error(`Coordinate transform mode ${coordinateTransferMode} is not supported`);
      }
    })() +
    '}';

const getNearestPixelFromOriginal = (nearestMode: NearestMode, opsetVersion: number): string =>
    'fn getNearestPixelFromOriginal(xOriginal: f32, isDownSample: bool) -> f32 {' + (() => {
      switch (nearestMode) {
        case 'round_prefer_ceil':
          return 'if (fract(xOriginal) == 0.5) { \
            return ceil(xOriginal); \
          } else { \
            return round(xOriginal); \
          }';
        case 'floor':
          return 'return floor(xOriginal);';
        case 'ceil':
          return 'return ceil(xOriginal);';
        case 'round_prefer_floor':
          return 'if (fract(xOriginal) == 0.5) { \
                    return floor(xOriginal); \
                  } else { \
                    return round(xOriginal); \
                  }';
        case 'simple':
        default:
          if (opsetVersion < 11) {
            return 'if (isDownSample) \
                    { \
                      return ceil(xOriginal); \
                    } else { \
                      return xOriginal; \
                    }';
          }
          throw new Error(`Nearest mode ${nearestMode} is not supported`);
      }
    })() +
    '}';

const updateRoI = (roi: readonly number[], axes: readonly number[], rank: number): number[] => {
  const roiTmp = new Array(rank).fill(0).concat(new Array(rank).fill(1));
  const roiLocal = roi.length === 0 ? roiTmp : roi.slice();
  if (axes.length > 0) {
    axes.forEach((v, i) => {
      roiTmp[v] = roiLocal[i];
      roiTmp[i + rank] = roiLocal[axes.length + i];
    });
    return roiTmp;
  }
  return roiLocal;
};

const initOutputShape =
    (inputShape: readonly number[], scales: readonly number[], sizes: readonly number[], axes: readonly number[]):
        number[] => {
          let outputShape: number[] = [];
          if (sizes.length > 0) {
            if (axes.length > 0) {
              inputShape.forEach((v) => outputShape.push(v));
              if (Math.max(...axes) > inputShape.length) {
                throw new Error('axes is out of bound');
              }
              axes.forEach((v, i) => outputShape[v] = sizes[i]);
            } else {
              sizes.forEach((v) => outputShape.push(v));
            }
          } else {
            if (scales.length === 0) {
              throw new Error('Resize requires either scales or sizes.');
            } else {
              outputShape = inputShape.map((value, index) => Math.round(value * scales[index]));
            }
          }
          return outputShape;
        };

const adjustOutputShape = (inputShape: readonly number[], scales: number[], attributes: ResizeAttributes): number[] => {
  const scaleInPolicy = (() => {
    switch (attributes.keepAspectRatioPolicy) {
      case 'not_larger':
        return attributes.axes.length > 0 ? Math.min(...attributes.axes.map(i => scales[i]), Number.MAX_VALUE) :
                                            Math.min(...scales, Number.MAX_VALUE);
      case 'not_smaller':
        return attributes.axes.length > 0 ? Math.max(...attributes.axes.map(i => scales[i]), Number.MIN_VALUE) :
                                            Math.max(...scales, Number.MIN_VALUE);
      default:
        throw new Error(`Keep aspect ratio policy ${attributes.keepAspectRatioPolicy} is not supported`);
    }
  })();
  scales.fill(1.0, 0, scales.length);
  const adjustedOutputShape = inputShape.slice();
  if (attributes.axes.length > 0) {
    attributes.axes.forEach((v) => scales[v] = scaleInPolicy);
    attributes.axes.forEach((v) => adjustedOutputShape[v] = Math.round(inputShape[v] * scales[v]));
  } else {
    scales.fill(scaleInPolicy, 0, scales.length);
    adjustedOutputShape.forEach((v, i) => adjustedOutputShape[i] = Math.round(v * scales[i]));
  }
  return adjustedOutputShape;
};

const calculateOriginalIndicesFromOutputIndices =
    (output: IndicesHelper, inputShape: readonly number[], outputShape: readonly number[], scales: readonly number[],
     roi: readonly number[]): string => `
    fn calculateOriginalIndicesFromOutputIndices(outputIndices: ${output.type.indices}) -> array<f32, ${
        outputShape.length}> {
      const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
      const outputShape = array<u32, ${outputShape.length}>(${outputShape.map(i => `${i}u`).join(',')});
      const scales = array<f32, ${scales.length}>(${scales.map(i => `${i}f`).join(',')});
      const roi = array<f32, ${roi.length}>(${roi.map(i => `${i}f`).join(',')});
      var originalIndices: array<f32, ${outputShape.length}>;
      for (var i:u32 = 0; i < ${outputShape.length}; i++) {
        var outputIndex = ${outputShape.length === 1 ? 'outputIndices' : 'outputIndices[i]'};
        if (scales[i] == 1.0) {
          originalIndices[i] = f32(outputIndex);
        } else {
          originalIndices[i] = getOriginalCoordinateFromResizedCoordinate(f32(outputIndex), scales[i],
                f32(outputShape[i]), f32(inputShape[i]), roi[i], roi[i + ${inputShape.length}]);
        }
      }
      return originalIndices;
    }`;

const calculateInputIndicesFromOutputIndices =
    (input: IndicesHelper, output: IndicesHelper, inputShape: readonly number[], outputShape: readonly number[],
     scales: readonly number[], roi: readonly number[], useExtrapolation: boolean): string => `
    fn calculateInputIndicesFromOutputIndices(outputIndices: ${output.type.indices}) -> ${input.type.indices} {
        const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
        const outputShape = array<u32, ${outputShape.length}>(${outputShape.map(i => `${i}u`).join(',')});
        const scales = array<f32, ${scales.length}>(${scales.map(i => `${i}f`).join(',')});
        const roi = array<f32, ${roi.length}>(${roi.map(i => `${i}f`).join(',')});
        var inputIndices: ${input.type.indices};
        for (var i:u32 = 0; i < ${outputShape.length}; i++) {
          var outputIndex = ${outputShape.length === 1 ? 'outputIndices' : 'outputIndices[i]'};
          var inputIndex: u32;
          if (scales[i] == 1.0) {
            inputIndex = outputIndex;
          } else {
            var original_idx = getOriginalCoordinateFromResizedCoordinate(f32(outputIndex), scales[i],
                    f32(outputShape[i]), f32(inputShape[i]), roi[i], roi[i + ${inputShape.length}]);
            if (!${useExtrapolation} || (original_idx >= 0 && original_idx < f32(inputShape[i]))) {
              if (original_idx < 0) {
                inputIndex = 0;
              } else if (original_idx > (f32(inputShape[i]) - 1)) {
                inputIndex = inputShape[i] - 1;
              } else {
                inputIndex = u32(getNearestPixelFromOriginal(original_idx, scales[i] < 1));
              }
            } else {
              inputIndex = u32(original_idx);
            }
          }
          ${input.indicesSet('inputIndices', 'i', 'inputIndex')}
        }
        return inputIndices;
    }`;

const checkInputIndices = (input: IndicesHelper, inputShape: readonly number[]): string => `
    fn checkInputIndices(inputIndices: ${input.type.indices}) -> bool {
      const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
      for (var i:u32 = 0; i < ${inputShape.length}; i++) {
        var inputIndex = ${inputShape.length === 1 ? 'inputIndices' : 'inputIndices[i]'};
        if (inputIndex < 0 || inputIndex >= inputShape[i]) {
          return false;
        }
      }
      return true;
    }`;

const bilinearInterpolation =
    (input: IndicesHelper, output: IndicesHelper, inputShape: readonly number[], scales: readonly number[],
     useExtrapolation: boolean, extrapolationValue: number): string => {
      const [batchIdx, heightIdx, widthIdx, channelIdx] =
          inputShape.length === 2 ? [-1, 0, 1, -1] : (scales[1] === 1.0 ? [0, 2, 3, 1] : [0, 1, 2, 3]);
      return `
    fn getInputValue(batch: u32, channel: u32, row: u32, col: u32) -> f32 {
      var inputIndices: ${input.type.indices};
      inputIndices[${heightIdx}] = max(0, min(row, ${inputShape[heightIdx]} - 1));
      inputIndices[${widthIdx}] = max(0, min(col, ${inputShape[widthIdx]} - 1));
      if (${inputShape.length} > 2) {
        inputIndices[${channelIdx}] = channel;
        inputIndices[${batchIdx}] = batch;
      };
      return input[${input.indicesToOffset('inputIndices')}];
    }

    fn bilinearInterpolation(outputIndices: ${output.type.indices}) -> f32 {
      var originalIndices = calculateOriginalIndicesFromOutputIndices(outputIndices);
      var row:f32 = originalIndices[${heightIdx}];
      var col:f32 = originalIndices[${widthIdx}];
      if (${useExtrapolation} && (row < 0 || row > (${inputShape[heightIdx]} - 1) || col < 0 || col > ${
          inputShape[widthIdx]} - 1)) {
        return ${extrapolationValue};
      }
      row = max(0, min(row, ${inputShape[heightIdx]} - 1));
      col = max(0, min(col, ${inputShape[widthIdx]} - 1));
      var row1: u32 = u32(row);
      var col1: u32 = u32(col);
      var row2: u32 = u32(row + 1);
      var col2: u32 = u32(col + 1);
      var channel: u32 = 0;
      var batch: u32 = 0;
      if (${inputShape.length > 2}) {
        channel = u32(originalIndices[${channelIdx}]);
        batch = u32(originalIndices[${batchIdx}]);
      }
      var x11: f32 = getInputValue(batch, channel, row1, col1);
      var x12: f32 = getInputValue(batch, channel, row1, col2);
      var x21: f32 = getInputValue(batch, channel, row2, col1);
      var x22: f32 = getInputValue(batch, channel, row2, col2);
      var dx1: f32 = row - f32(row1);
      var dx2: f32 = f32(row2 ) - row;
      var dy1 = col - f32(col1);
      var dy2 = f32(col2) - col;
      return (x11 * dx2 * dy2 + x12 * dx2 * dy1 + x21 * dx1 * dy2 + x22 * dx1 * dy1);
    }`;
    };

const bicubicInterpolation =
    (input: IndicesHelper, output: IndicesHelper, inputShape: readonly number[], outputShape: readonly number[],
     scales: readonly number[], roi: readonly number[], cubicCoeffA: number, useExtrapolation: boolean,
     extrapolationValue: number, excludeOutside: boolean): string => {
      const [heightIdx, widthIdx] = inputShape.length === 2 ? [0, 1] : (scales[1] === 1.0) ? [2, 3] : [1, 2];

      const createCubicInterpolationFunction = (idx: number): string => {
        const direction = idx === heightIdx ? 'row' : 'col';
        return `
      fn ${direction}CubicInterpolation(inputIndices: ${input.type.indices}, outputIndices: ${
            output.type.indices}) -> f32 {
        var outputIndex = ${outputShape.length === 1 ? 'outputIndices' : `outputIndices[${idx}]`};
        var originalIdx: f32 = getOriginalCoordinateFromResizedCoordinate(f32(outputIndex), ${scales[idx]},
        f32(${outputShape[idx]}), f32(${inputShape[idx]}), ${roi[idx]}, ${roi[idx]} + ${inputShape.length});
        var fractOriginalIdx: f32 = originalIdx - floor(originalIdx);
        var coefs = getCubicInterpolationCoefs(fractOriginalIdx);

        if (${useExtrapolation} && (originalIdx < 0 || originalIdx > (${inputShape[idx]} - 1))) {
          return ${extrapolationValue};
        }
        var data: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        for (var i: i32 = -1; i < 3; i++) {
          var ${direction}: f32 = originalIdx + f32(i);
          if (${direction} < 0 || ${direction} >= ${inputShape[idx]}) {
            if (${excludeOutside}) {
              coefs[i + 1] = 0.0;
              continue;
            } else if (${useExtrapolation}) {
              return ${extrapolationValue};
            } else {
              ${direction} = max(0, min(${direction}, ${inputShape[idx]} - 1));
            }
          }
          var inputIndicesCopy: ${input.type.indices} = inputIndices;
          inputIndicesCopy[${idx}] = u32(${direction});
          data[i + 1] = ${idx === heightIdx ? `input[${input.indicesToOffset('inputIndicesCopy')}];` : `
                                               rowCubicInterpolation(inputIndicesCopy, outputIndices);`}
        }
        return cubicInterpolation1D(data, coefs);
      }`;
      };

      return `
    ${createCubicInterpolationFunction(heightIdx)};
    ${createCubicInterpolationFunction(widthIdx)};
  fn getCubicInterpolationCoefs(s: f32) -> array<f32, 4> {
    var absS = abs(s);
    var coeffs: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var oneMinusAbsS: f32 = 1.0 - absS;
    var twoMinusAbsS: f32 = 2.0 - absS;
    var onePlusAbsS: f32 = 1.0 + absS;
    coeffs[0] = ((${cubicCoeffA} * onePlusAbsS - 5 * ${cubicCoeffA}) * onePlusAbsS + 8 * ${
          cubicCoeffA}) * onePlusAbsS - 4 * ${cubicCoeffA};
    coeffs[1] = ((${cubicCoeffA} + 2) * absS - (${cubicCoeffA} + 3)) * absS * absS + 1;
    coeffs[2] = ((${cubicCoeffA} + 2) * oneMinusAbsS - (${cubicCoeffA} + 3)) * oneMinusAbsS * oneMinusAbsS + 1;
    coeffs[3] = ((${cubicCoeffA} * twoMinusAbsS - 5 * ${cubicCoeffA}) * twoMinusAbsS + 8 * ${
          cubicCoeffA}) * twoMinusAbsS - 4 * ${cubicCoeffA};
    return coeffs;
  }

  fn cubicInterpolation1D(x: array<f32, 4>, coefs: array<f32, 4>) -> f32 {
    var coefsSum: f32 = coefs[0] + coefs[1] + coefs[2] + coefs[3];
    return (x[0] * coefs[0] + x[1] * coefs[1]+ x[2] * coefs[2]+ x[3] * coefs[3]) / coefsSum;
  }

  fn bicubicInterpolation(outputIndices: ${output.type.indices}) -> f32 {
    var inputIndices: ${input.type.indices} = outputIndices;
    return colCubicInterpolation(inputIndices, outputIndices);
  }
    `;
    };

const createResizeProgramInfo =
    (inputTensor: TensorView, attributes: ResizeAttributes, opsetVersion: number, scalesInput: readonly number[],
     sizes: readonly number[], roiInput: readonly number[]): ProgramInfo => {
      const inputShape = inputTensor.dims;
      const roi = updateRoI(roiInput, attributes.axes, inputShape.length);

      let outputShape = initOutputShape(inputShape, scalesInput, sizes, attributes.axes);
      let scales = scalesInput.slice();
      if (scalesInput.length === 0) {
        scales = inputShape.map((value, index) => value === 0 ? 1.0 : outputShape[index] / value);
        if (attributes.keepAspectRatioPolicy !== 'stretch') {
          outputShape = adjustOutputShape(inputShape, scales, attributes);
        }
      }
      const output = outputVariable('output', inputTensor.dataType, outputShape);
      const input = inputVariable('input', inputTensor.dataType, inputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const noScale = inputShape.length === outputShape.length && inputShape.every((d, i) => d === outputShape[i]);
      const useExtrapolation = attributes.coordinateTransformMode === 'tf_crop_and_resize';
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${noScale ? '' : `
      ${getOriginalCoordinateFromResizedCoordinate(attributes.coordinateTransformMode)};
      ${(() => {
        switch (attributes.mode) {
          case 'nearest':
            return `
              ${checkInputIndices(input, inputShape)};
              ${getNearestPixelFromOriginal(attributes.nearestMode, opsetVersion)};
              ${
                calculateInputIndicesFromOutputIndices(
                    input, output, inputShape, outputShape, scales, roi, useExtrapolation)};
              `;
          case 'linear':
            return `
              ${calculateOriginalIndicesFromOutputIndices(output, inputShape, outputShape, scales, roi)};
              ${
                bilinearInterpolation(
                    input, output, inputShape, scales, useExtrapolation, attributes.extrapolationValue)};
              `;
          case 'cubic':
            return `
            ${
                bicubicInterpolation(
                    input, output, inputShape, outputShape, scales, roi, attributes.cubicCoeffA, useExtrapolation,
                    attributes.extrapolationValue, attributes.excludeOutside)};
            `;
          default:
            throw Error('Invalid resize mode');
        }
      })()};
      `}
      ${shaderHelper.declareVariables(input, output)}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        ${noScale ? 'output[global_idx] = input[global_idx];' : `
        let outputIndices = ${output.offsetToIndices('global_idx')};
        var inputIndices: ${input.type.indices};
        ${(() => {
        switch (attributes.mode) {
          case 'nearest':
            return `inputIndices = calculateInputIndicesFromOutputIndices(outputIndices);
                if (checkInputIndices(inputIndices)) {
                  output[global_idx] = input[${input.indicesToOffset('inputIndices')}];
                } else {
                  output[global_idx] = ${attributes.extrapolationValue};
                }`;
          case 'linear':
            return 'output[global_idx] = bilinearInterpolation(outputIndices);';
          case 'cubic':
            return 'output[global_idx] = bicubicInterpolation(outputIndices);';
          default:
            throw Error(`Unsupported resize mode: ${attributes.mode}`);
        }
      })()};
        `}
      }`;

      return {
        name: 'Resize',
        shaderCache: {
          hint: `${attributes.cacheKey}|${opsetVersion}|${scales.length > 0 ? scales : ''}|${
              sizes.length > 0 ? sizes : ''}|${noScale}`
        },
        getShaderSource,
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputTensor.dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}
        })
      };
    };

const getOpsetVersionFromCustomDataBuffer = (context: ComputeContext): number => {
  const customDataBuffer = context.customDataBuffer;
  const customDataBuffer32 = new Uint32Array(customDataBuffer, customDataBuffer.byteOffset, 1);
  const opsetVersion = customDataBuffer32[0];
  return opsetVersion;
};

export const resize = (context: ComputeContext, attributes: ResizeAttributes): void => {
  const scales: number[] = [];
  const sizes: number[] = [];
  const roi: number[] = [];
  const opsetVersion = getOpsetVersionFromCustomDataBuffer(context);
  validateInputs(context.inputs, attributes, opsetVersion, scales, sizes, roi);
  context.compute(
      createResizeProgramInfo(context.inputs[0], attributes, opsetVersion, scales, sizes, roi), {inputs: [0]});
};

export const parseResizeAttributes = (attributes: Record<string, unknown>): ResizeAttributes => {
  const antialias = attributes.antialias as number;
  const axes = attributes.axes as number[];
  const coordinateTransformMode: CoordinateTransformMode =
      attributes.coordinateTransformMode as CoordinateTransformMode;
  const cubicCoeffA = attributes.cubicCoeffA as number;
  const excludeOutside = attributes.excludeOutside as number !== 0;
  const extrapolationValue = attributes.extrapolationValue as number;
  const keepAspectRatioPolicy: KeepAspectRatioPolicy = attributes.keepAspectRatioPolicy as KeepAspectRatioPolicy;
  const mode: Mode = attributes.mode as Mode;
  // If nearestMode is not specified, use simple mode.
  const nearestMode: NearestMode = (attributes.nearestMode === '' ? 'simple' : attributes.nearestMode) as NearestMode;
  return createAttributeWithCacheKey({
    antialias,
    axes,
    coordinateTransformMode,
    cubicCoeffA,
    excludeOutside,
    extrapolationValue,
    keepAspectRatioPolicy,
    mode,
    nearestMode
  });
};
