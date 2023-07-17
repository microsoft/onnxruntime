// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

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

const validateScales = (scales: number[]): void => {
  scales.every((value) => value > 0 || (() => {
                            throw new Error('Resize requires scales input values to be positive');
                          }));
  // Check scales dims based on mode: LINEAR, CUBIC
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
        validateScales(scales);
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
                    return roi_start * (lengthOriginal - 1) + \
                          (xResized * (roiEnd - roiStart) * (lengthResized - 1)) / (lengthResized - 1); \
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
          return 'return round(xOriginal);';
        case 'floor':
          return 'return floor(xOriginal);';
        case 'ceil':
          return 'return ceil(xOriginal);';
        case 'round_prefer_floor':
          return 'if (xOriginal == floor(xOriginal) + 0.5)) { \
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

const adjustOutputShape =
    (inputShape: readonly number[], outputShape: readonly number[], scales: number[], attributes: ResizeAttributes):
        number[] => {
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
    (inputShape: readonly number[], outputShape: readonly number[], scales: readonly number[], roi: readonly number[]):
        string => `
     fn calculateOriginalIndicesFromOutputIndices(outputIndices: array<u32, ${outputShape.length}>) -> array<f32, ${
            outputShape.length}> {
      const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
      const outputShape = array<u32, ${outputShape.length}>(${outputShape.map(i => `${i}u`).join(',')});
      const scales = array<f32, ${scales.length}>(${scales.map(i => `${i}f`).join(',')});
      const roi = array<f32, ${roi.length}>(${roi.map(i => `${i}f`).join(',')});
      var originalIndices: array<f32, ${outputShape.length}>;
      for (var i:u32 = 0; i < ${outputShape.length}; i++) {
        var original_idx = getOriginalCoordinateFromResizedCoordinate(f32(outputIndices[i]), scales[i],
                 f32(outputShape[i]), f32(inputShape[i]), roi[i], roi[i + ${inputShape.length}]);
        if (original_idx < 0) {
          original_idx = 0.0;
        } else if (original_idx > (f32(inputShape[i]) - 1.0)) {
          original_idx = f32(inputShape[i]) - 1.0;
        }
        originalIndices[i] = original_idx;
      }
      return originalIndices;
    }`;

const calculateInputIndicesFromOutputIndices =
    (inputShape: readonly number[], outputShape: readonly number[], scales: readonly number[], roi: readonly number[],
     useExtrapolation: boolean): string => `
     fn calculateInputIndicesFromOutputIndices(outputIndices: array<u32, ${outputShape.length}>) -> array<u32, ${
        inputShape.length}> {
          const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
          const outputShape = array<u32, ${outputShape.length}>(${outputShape.map(i => `${i}u`).join(',')});
          const scales = array<f32, ${scales.length}>(${scales.map(i => `${i}f`).join(',')});
          const roi = array<f32, ${roi.length}>(${roi.map(i => `${i}f`).join(',')});
          var inputIndices: array<u32, ${inputShape.length}>;
          for (var i:u32 = 0; i < ${outputShape.length}; i++) {
            var original_idx = getOriginalCoordinateFromResizedCoordinate(f32(outputIndices[i]), scales[i],
                     f32(outputShape[i]), f32(inputShape[i]), roi[i], roi[i + ${inputShape.length}]);
            if (!${useExtrapolation} || (original_idx >= 0 && original_idx < f32(inputShape[i]))) {
              if (original_idx < 0) {
                inputIndices[i] = 0;
              } else if (original_idx > (f32(inputShape[i]) - 1)) {
                inputIndices[i] = inputShape[i] - 1;
              } else {
                inputIndices[i] = u32(getNearestPixelFromOriginal(original_idx, scales[i] < 1));
              }
            } else {
              inputIndices[i] = u32(original_idx);
            }
          }
          return inputIndices;
     }`;

const checkInputIndices = (inputShape: readonly number[]): string => `
    fn checkInputIndices(inputIndices: array<u32, ${inputShape.length}>) -> bool {
      const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});
      for (var i:u32 = 0; i < ${inputShape.length}; i++) {
        if (inputIndices[i] < 0 || inputIndices[i] >= inputShape[i]) {
          return false;
        }
      }
      return true;
    }`;

const bilinearInterpolation =
    (inputShape: readonly number[], outputShape: readonly number[], scales: readonly number[],
     useExtrapolation: boolean, extrapolationValue: number): string => {
      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const [heightIdx, widthIdx] = inputShape.length === 2 ? [0, 1] : (scales[1] === 1.0) ? [2, 3] : [1, 2];
      return `
    fn bilinearInterpolation(outputIndices: ${outputIndicesHelper.iType}) -> f32 {
      var inputIndicesX11: ${inputIndicesHelper.iType} = outputIndices;
      var inputIndicesX12: ${inputIndicesHelper.iType} = outputIndices;
      var inputIndicesX21: ${inputIndicesHelper.iType} = outputIndices;
      var inputIndicesX22: ${inputIndicesHelper.iType} = outputIndices;
      var originalIndices = calculateOriginalIndicesFromOutputIndices(outputIndices);
      var row:f32 = originalIndices[${heightIdx}];
      var col:f32 = originalIndices[${widthIdx}];
      var row1: u32 = u32(row);
      var col1: u32 = u32(col);
      var row2: u32 = u32(row + 1);
      var col2: u32 = u32(col + 1);
      if (!${useExtrapolation}) {
        if (row2 > ${inputShape[heightIdx]}) {
          row2 = ${inputShape[heightIdx]} - 1;
        }
        if (col2 > ${inputShape[widthIdx]}) {
          col2 = ${inputShape[widthIdx]} - 1;
        }
      } else {
        if (row > ${inputShape[heightIdx]} - 1 || row < 0 || col > ${inputShape[widthIdx]} - 1 || col < 0) {
          return ${extrapolationValue};
        }
      }
      inputIndicesX11[${heightIdx}] = row1;
      inputIndicesX11[${widthIdx}] = col1;
      inputIndicesX12[${heightIdx}] = row1;
      inputIndicesX12[${widthIdx}] = col2;
      inputIndicesX21[${heightIdx}] = row2;
      inputIndicesX21[${widthIdx}] = col1;
      inputIndicesX22[${heightIdx}] = row2;
      inputIndicesX22[${widthIdx}] = col2;
      var x11: f32 = input[${inputIndicesHelper.i2oExpression('inputIndicesX11')}];
      var x12: f32 = input[${inputIndicesHelper.i2oExpression('inputIndicesX12')}];
      var x21: f32 = input[${inputIndicesHelper.i2oExpression('inputIndicesX21')}];
      var x22: f32 = input[${inputIndicesHelper.i2oExpression('inputIndicesX22')}];
      var dx1: f32 = row - f32(row1);
      var dx2: f32 = f32(row2 ) - row;
      var dy1 = col - f32(col1);
      var dy2 = f32(col2) - col;
      return (x11 * dx2 * dy2 + x12 * dx2 * dy1 + x21 * dx1 * dy2 + x22 * dx1 * dy1);
    }`;
    };
const bicubicInterpolation =
    (inputShape: readonly number[], outputShape: readonly number[], scales: readonly number[], cubicCoeffA: number,
     useExtrapolation: boolean, extrapolationValue: number, excludeOutside: boolean): string => {
      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const [heightIdx, widthIdx] = inputShape.length === 2 ? [0, 1] : (scales[1] === 1.0) ? [2, 3] : [1, 2];
      return `
  fn getCubicInterpolationCoefs(s: f32) -> array<f32, 4> {
    absS = abs(s);
    var coeffs: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var oneMinusAbsS: f32 = 1.0 - absS;
    var twoMinusAbsS: f32 = 2.0 - absS;
    var onePlusAbsS: f32 = 1.0 + absS;
    coeffs[0] = ((${cubicCoeffA} * onePlusAbsS - 5 * ${cubicCoeffA}) * onePlusAbsS + 8 * ${
          cubicCoeffA}) * onePlusAbsS - 4 * ${cubicCoeffA};
    coeffs[1] = ((${cubicCoeffA} + 2) * absS - (${cubicCoeffA} + 3)) * absS * onePlusAbsS;
    coeffs[2] = ((${cubicCoeffA} + 2) * oneMinusAbsS - (${cubicCoeffA} + 3)) * oneMinusAbsS * oneMinusAbsS + 1;
    coeffs[3] = ((${cubicCoeffA} * twoMinusAbsS - 5 * ${cubicCoeffA}) * twoMinusAbsS + 8 * ${
          cubicCoeffA}) * twoMinusAbsS - 4 * ${cubicCoeffA};
    return coeffs;
  }

  fn cubicInterpolation1D(x: array<f32, 4>, coeffs: array<f32, 4>) -> f32 {
    var coefsSum: f32 = coefs[0] + coefs[1] + coefs[2] + coefs[3];
    return dot(x, coeffs/coefsSum);
  }

  fn bicubicInterpolation(outputIndices: ${outputIndicesHelper.iType}) -> f32 {
    var originalIndices = calculateOriginalIndicesFromOutputIndices(outputIndices);
    var originRow:f32 = originalIndices[${heightIdx}];
    var originCol:f32 = originalIndices[${widthIdx}];
    colCoefs = getCubicInterpolationCoefs(fract(originCol));
    rowCoefs = getCubicInterpolationCoefs(fract(originRow));

    var colData: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var c: i32 = -1; c < 3; c++) {
      var col: i32 = i32(originCol) + c;
      if (col < 0 || col >= ${inputShape[widthIdx]}) {
        if (${excludeOutside}) {
          colCoefs[r + 1] = 0.0;
          colData[r + 1] = 0.0;
          continue;
        } else if (${useExtrapolation}) {
          colData[r + 1] = ${extrapolationValue};
        } else {
          if (col < 0) {
            col = 0;
          } else {
            col = ${inputShape[widthIdx]} - 1;
          }
        }
      }
      var rowData: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
      var rowCoefsCopy = rowCoefs;
      for (var r: i32 = -1; r < 3; r++) {
        var row: i32 = i32(originRow) + r;
        if (row < 0 || row >= ${inputShape[heightIdx]}) {
          if (${excludeOutside}) {
            rowCoefsCopy[r + 1] = 0.0;
            rowData[r + 1] = 0.0;
            continue;
          } else if (${useExtrapolation}) {
            rowData[r + 1] = ${extrapolationValue};
          } else {
            if (row < 0) {
              row = 0;
            } else {
              row = ${inputShape[heightIdx]} - 1;
            }
          }
        }
        var inputIndices: ${inputIndicesHelper.iType} = outputIndices;
        inputIndices[${heightIdx}] = row;
        inputIndices[${widthIdx}] = col;
        rowData[r + 1] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
      }
      colData[c + 1] = cubicInterpolation1D(rowData, rowCoefsCopy);
    }
    return cubicInterpolation1D(colData, colCoefs);
  }
    `;
    };

const createResizeProgramInfo =
    (metadata: ProgramMetadata, input: TensorView, attributes: ResizeAttributes, opsetVersion: number,
     scalesInput: readonly number[], sizes: readonly number[], roiInput: readonly number[]): ProgramInfo => {
      const inputShape = input.dims;
      const roi = updateRoI(roiInput, attributes.axes, inputShape.length);

      let outputShape = initOutputShape(inputShape, scalesInput, sizes, attributes.axes);
      let scales = scalesInput.slice();
      if (scalesInput.length === 0) {
        scales = inputShape.map((value, index) => value === 0 ? 1.0 : outputShape[index] / value);
        if (attributes.keepAspectRatioPolicy !== 'stretch') {
          outputShape = adjustOutputShape(inputShape, outputShape, scales, attributes);
        }
      }
      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const dataType = 'f32';
      const noScale = inputShape.length === outputShape.length && inputShape.every((d, i) => d === outputShape[i]);
      const useExtrapolation = attributes.coordinateTransformMode === 'tf_crop_and_resize';
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${getOriginalCoordinateFromResizedCoordinate(attributes.coordinateTransformMode)};
      ${(() => {
        switch (attributes.mode) {
          case 'nearest':
            return `
              ${checkInputIndices(inputShape)};
              ${getNearestPixelFromOriginal(attributes.nearestMode, opsetVersion)};
              ${calculateInputIndicesFromOutputIndices(inputShape, outputShape, scales, roi, useExtrapolation)};
              `;
          case 'linear':
            return `
              ${calculateOriginalIndicesFromOutputIndices(inputShape, outputShape, scales, roi)};
              ${
                bilinearInterpolation(
                    inputShape, outputShape, scales, useExtrapolation, attributes.extrapolationValue)};
              `;
          case 'cubic':
            return `
            ${calculateOriginalIndicesFromOutputIndices(inputShape, outputShape, scales, roi)};
            ${
                bicubicInterpolation(
                    inputShape, outputShape, scales, attributes.cubicCoeffA, useExtrapolation,
                    attributes.extrapolationValue, attributes.excludeOutside)};
            `;
          default:
            throw Error('Invalid resize mode');
        }
      })()};
      @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
      @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;
      ${outputIndicesHelper.o2iImpl}
      ${inputIndicesHelper.i2oImpl}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        if (${noScale}) {
          output[global_idx] = input[global_idx];
        } else {
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          ${(() => {
        switch (attributes.mode) {
          case 'nearest':
            return `  inputIndices = calculateInputIndicesFromOutputIndices(outputIndices);
                      if (checkInputIndices(inputIndices)) {
                        output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
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
        }
      }`;

      return {
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType: input.dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const createResizeProgramInfoLoader =
    (input: TensorView, attributes: ResizeAttributes, opsetVersion: number, scales: readonly number[],
     sizes: readonly number[], roi: readonly number[]): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {
        name: 'Resize',
        inputTypes: [GpuDataType.default],
        cacheHint: attributes.cacheKey + opsetVersion.toString() + (scales.length > 0 ? '_scales' : '') +
            (sizes.length > 0 ? '_sizes' : ''),
      };
      return {
        ...metadata,
        get: () => createResizeProgramInfo(metadata, input, attributes, opsetVersion, scales, sizes, roi)
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
      createResizeProgramInfoLoader(context.inputs[0], attributes, opsetVersion, scales, sizes, roi), {inputs: [0]});
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
