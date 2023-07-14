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
  excludeOutsize: number;
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
            (scales.length !== rank && (opsetVersion >= 18 && scales.length === attributes.axes.length))) {
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
    'fn getOriginalCoordinateFromResizedCoordinate(xResized: u32, xScale: f32, lengthResized: u32,\
    llengthOriginal: u32, roiStart: f32, roiEnd: f32) -> f32 { ' +
    (() => {
      switch (coordinateTransferMode) {
        case 'asymmetric':
          return 'return f32(xResized) / xScale;';
        case 'pytorch_half_pixel':
          return 'return lengthResized > 1 ? (f32(xResized) + 0.5) / xScale - 0.5 : 0.0;';
        case 'tf_half_pixel_for_nn':
          return 'return (f32(xResized) + 0.5) / xScale';
        case 'align_corners':
          return 'return lengthResized === 1 ? 0 : xResized * (llengthOriginal - 1) / (lengthResized - 1);';
        case 'tf_crop_and_resize':
          return 'return lengthResized > 1 ? \
          (xResized * (roiEnd - roiStart) / (lengthResized - 1)) / (lengthResized - 1) : \
          0.5 * (roiStart + roiEnd) * (llengthOriginal - 1);';
        case 'half_pixel_symmetric':
          return [
            'const outputWidth = xScale * lengthResized;', 'const adjustment = lengthResized / outputWidth;',
            'const center = llengthOriginal / 2;', 'const offset = center * (1 - adjustment);',
            'return offset + (f32(xResized) + 0.5) / xScale - 0.5;'
          ].join('\n');
        case 'half_pixel':
          return 'return (f32(xResized) + 0.5) / xScale - 0.5;';
        default:
          throw new Error(`Coordinate transform mode ${coordinateTransferMode} is not supported`);
      }
    })() +
    '}';

const getNearestPixelFromOriginal = (nearestMode: NearestMode): string =>
    'fn getNearestPixelFromOriginal(xOriginal: f32, isDownSample: bool) -> f32 {' + (() => {
      switch (nearestMode) {
        case 'simple':
          return 'if (isDownSample)\n {\nreturn ceil(xOriginal);\n} else {\n return xOriginal;\n}';
        case 'round_prefer_ceil':
          return 'return round(xOriginal);';
        case 'floor':
          return 'return floor(xOriginal);';
        case 'ceil':
          return 'return ceil(xOriginal);';
        case 'round_prefer_floor':
          return 'if (xOriginal == roudnd(xOriginal) + 0.5) {return floor(xOriginal);} else {return round(xOriginal);}';
        default:
          throw new Error(`Nearest mode ${nearestMode} is not supported`);
      }
    })() +
    '}';

const updateRoI = (roi: readonly number[], axes: readonly number[], rank: number): number[] => {
  const roiTmp = new Array(rank).fill(0).concat(new Array(rank).fill(1));
  let roiLocal = roi.slice();
  if (roi.length === 0) {
    roiLocal = roiTmp;
  }
  if (axes.length > 0) {
    axes.forEach((v, i) => {
      roiTmp[v] = roiLocal[i];
      roiTmp[i + rank] = roiLocal[axes.length + i];
    });
    roiLocal = roiTmp;
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
    (inputShape: readonly number[], outputShape: readonly number[], attributes: ResizeAttributes): number[] => {
      const scales = inputShape.map((value, index) => value === 0 ? 1.0 : outputShape[index] / value);
      if (attributes.keepAspectRatioPolicy !== 'stretch') {
        const scaleInPolicy = (() => {
          switch (attributes.keepAspectRatioPolicy) {
            case 'not_larger':
              return Math.min(...scales);
            case 'not_smaller':
              return Math.max(
                  attributes.axes.length > 0 ? Math.min(...attributes.axes.map(i => scales[i])) : Math.min(...scales));
            default:
              throw new Error(`Keep aspect ratio policy ${attributes.keepAspectRatioPolicy} is not supported`);
          }
        })();
        scales.fill(1.0, 0, scales.length);
        const adjustedOutputShape = inputShape.slice();
        if (attributes.axes.length > 0) {
          attributes.axes.forEach((v, i) => adjustedOutputShape[v] = Math.round(inputShape[v] * scales[i]));
          attributes.axes.forEach((v) => scales[v] = scaleInPolicy);
        }
        return adjustedOutputShape;
      } else {
        return outputShape.slice();
      }
    };

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
            var original_idx = getOriginalCoordinateFromResizedCoordinate(outputIndices[i], scales[i],
                     outputShape[i], inputShape[i], roi[i], roi[i + ${inputShape.length}]);
            if (!${useExtrapolation} || (original_idx >= 0 && original_idx < f32(inputShape[i]))) {
              if (original_idx < 0) {
                inputIndices[i] = 0;
              } else if (original_idx >= f32(inputShape[i])) {
                inputIndices[i] = inputShape[i] - 1;
              } else {
                inputIndices[i] = u32(getNearestPixelFromOriginal(original_idx, scales[i] < 1));
              }
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

const createResizeProgramInfo =
    (metadata: ProgramMetadata, input: TensorView, attributes: ResizeAttributes, opsetVersion: number,
     scales: readonly number[], sizes: readonly number[], roiInput: readonly number[]): ProgramInfo => {
      const inputShape = input.dims;
      const roi = updateRoI(roiInput, attributes.axes, inputShape.length);

      let outputShape = initOutputShape(inputShape, scales, sizes, attributes.axes);
      if (scales.length === 0) {
        outputShape = adjustOutputShape(inputShape, outputShape, attributes);
      }
      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const dataType = 'f32';
      const noScale = inputShape.length === outputShape.length && inputShape.every((d, i) => d === outputShape[i]);
      const useNearest2xOptimization = ((opsetVersion < 11) ? true :
                                                              (attributes.mode === 'nearest' &&
                                                               attributes.coordinateTransformMode === 'asymmetric' &&
                                                               attributes.nearestMode === 'floor')) &&
          scales.toString() === '1,1,2,2';
      const useNearest2xOptimizationSnippet = `
        inputIndices[0] = outputIndices[0]; // batch size
        inputIndices[1] = outputIndices[1]; // channel
        inputIndices[2] = outputIndices[2] / 2; // height
        inputIndices[3] = outputIndices[3] / 2; // width
      `;
      const useExtrapolation = attributes.coordinateTransformMode === 'tf_crop_and_resize';
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${attributes.mode === 'nearest' ? getNearestPixelFromOriginal(attributes.nearestMode) : ';'};
      ${checkInputIndices(inputShape)};
      ${calculateInputIndicesFromOutputIndices(inputShape, outputShape, scales, roi, useExtrapolation)};
      ${getOriginalCoordinateFromResizedCoordinate(attributes.coordinateTransformMode)};
      @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
      @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;
      ${outputIndicesHelper.o2iImpl}
      ${inputIndicesHelper.i2oImpl}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        if (${noScale}) {
          output[global_idx] = input[global_idx];
        }
        else
        {
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          if (${useNearest2xOptimization}) {
            ${useNearest2xOptimizationSnippet}
          } else {
            inputIndices = calculateInputIndicesFromOutputIndices(outputIndices);
            if (checkInputIndices(inputIndices)) {
               output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
            } else {
              output[global_idx] = ${attributes.extrapolationValue};
            }
          }
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
        cacheHint: attributes.cacheKey,
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
  const excludeOutsize = attributes.excludeOutsize as number;
  const extrapolationValue = attributes.extrapolationValue as number;
  const keepAspectRatioPolicy: KeepAspectRatioPolicy = attributes.keepAspectRatioPolicy as KeepAspectRatioPolicy;
  const mode: Mode = attributes.mode as Mode;
  // If nearestMode is not specified, use simple mode.
  const nearestMode: NearestMode = attributes.nearestMode === '' ? 'simple' : attributes.nearestMode as NearestMode;
  return createAttributeWithCacheKey({
    antialias,
    axes,
    coordinateTransformMode,
    cubicCoeffA,
    excludeOutsize,
    extrapolationValue,
    keepAspectRatioPolicy,
    mode,
    nearestMode
  });
};
