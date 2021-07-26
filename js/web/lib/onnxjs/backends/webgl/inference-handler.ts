// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {createPackProgramInfo} from './ops/pack';

import {encodeAsUint8} from './ops/uint8-encode';
import {createUnpackProgramInfo} from './ops/unpack';
import {WebGLSessionHandler} from './session-handler';
import {Encoder} from './texture-data-encoder';
// eslint-disable-next-line max-len
import {calculateTextureWidthAndHeight, createTextureLayoutFromShape, createTextureLayoutFromTextureType} from './texture-layout';
import {Artifact, ProgramInfo, TextureData, TextureLayout, TextureType} from './types';

const getProgramInfoUniqueKey = (programInfo: ProgramInfo, inputTextureDatas: TextureData[]): string => {
  const inputs =
      inputTextureDatas.map(texture => `${texture.unpackedShape.join(',')};${texture.width}x${texture.height}`)
          .join('_');
  let key = programInfo.name ?? '';
  key += '_' + inputs + '_' + programInfo.shaderSource;
  return key;
};

export class WebGLInferenceHandler implements InferenceHandler {
  private packedTextureDataCache: Map<Tensor.Id, TextureData>;
  private unpackedTextureDataCache: Map<Tensor.Id, TextureData>;
  constructor(public session: WebGLSessionHandler) {
    this.packedTextureDataCache = new Map();
    this.unpackedTextureDataCache = new Map();
  }

  /**
   * @returns [width, height]
   */
  calculateTextureWidthAndHeight(shape: readonly number[], textureType: TextureType): [number, number] {
    return calculateTextureWidthAndHeight(this.session.layoutStrategy, shape, textureType);
  }

  executeProgram(programInfo: ProgramInfo, inputs: readonly Tensor[]): TextureData {
    if (inputs.length < programInfo.inputNames.length) {
      throw new Error(`Input size mustn't be less than ${programInfo.inputNames.length}.`);
    }
    if (programInfo.inputNames.length !== programInfo.inputTypes.length) {
      throw new Error('input names size does not match input types');
    }

    // create texture info for input
    const inputTextureDatas: TextureData[] = [];
    for (let i = 0; i < programInfo.inputNames.length; ++i) {
      inputTextureDatas[i] = this.getOrCreateTextureData(inputs[i], programInfo.inputTypes[i]);
    }

    // create texture info for output
    const outputTextureLayout = createTextureLayoutFromTextureType(
        this.session.layoutStrategy, programInfo.output.dims, programInfo.output.textureType);
    const outputTextureData = this.createTextureData(outputTextureLayout, programInfo.output.type);

    const key = getProgramInfoUniqueKey(programInfo, inputTextureDatas);
    let artifact = this.session.programManager.getArtifact(key);
    if (!artifact) {
      artifact = this.session.programManager.build(programInfo, inputTextureDatas, outputTextureData);
      this.session.programManager.setArtifact(key, artifact);
    }

    this.runProgram(artifact, inputTextureDatas, outputTextureData);
    return outputTextureData;
  }

  run(programInfo: ProgramInfo, inputs: readonly Tensor[]): Tensor {
    const outputTextureData = this.executeProgram(programInfo, inputs);
    return outputTextureData.tensor;
  }

  private checkAndUpdateTextureForm(artifact: Artifact, inputs: TextureData[]) {
    // pack/unpack inputs
    for (let i = 0; i < inputs.length; ++i) {
      const input = inputs[i];
      if (input.isPacked && artifact.programInfo.inputTypes[i] !== TextureType.packed) {
        inputs[i] = this.unpack(input);
      } else if (!input.isPacked && artifact.programInfo.inputTypes[i] === TextureType.packed) {
        inputs[i] = this.pack(input);
      }
    }
  }
  private runProgram(artifact: Artifact, inputs: TextureData[], output: TextureData): void {
    this.checkAndUpdateTextureForm(artifact, inputs);

    // output should match
    if (!!output.isPacked !== (artifact.programInfo.output.textureType === TextureType.packed)) {
      throw new Error('output property packed inconsistent');
    }

    this.session.programManager.run(artifact, inputs, output);
  }

  /**
   * Create a TextureData object from a tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * If a related texture data is found in cache, returns it;
   * Otherwise:
   *   Creates a new texture layout if not provided;
   *   Creates WebGLTexture with the layout;
   *   Upload tensor data to the texture;
   *   Creates a texture data object associated with the given tensor.
   * @param tensor the tensor with data to upload
   */
  private getOrCreateTextureData(tensor: Tensor, textureType: TextureType) {
    let td = this.getTextureData(tensor.dataId, textureType === TextureType.packed);
    if (!td) {
      const layout = createTextureLayoutFromTextureType(this.session.layoutStrategy, tensor.dims, textureType);

      if (textureType === TextureType.packedLastDimension) {
        const group = 1;
        const channels = 4;
        const shape = tensor.dims;
        if (shape.length === 4) {
          // pre-processing for kernel data of Conv.
          //
          // TODO: currently this is a hacking to overwrite Conv's weight. The correct way to do this should be:
          // 1. implement texture based const-folding
          // 2. create a WebGL program "preprocessConvWeight" to do the same work as below
          // 3. run the program before dotProduct.
          //
          const adjustedKernelShape = [shape[0], Math.ceil((shape[1] * shape[2] * shape[3]) / channels)];
          const adjustedLayout =
              createTextureLayoutFromTextureType(this.session.layoutStrategy, adjustedKernelShape, textureType);
          let buffer = tensor.numberData;
          if (shape[2] * shape[3] % channels !== 0) {
            const numFeatureMaps = shape[0];
            const oldRowSize = shape[1] * shape[2] * shape[3];
            const newRowSize = Math.ceil(oldRowSize * group / channels) * channels;
            const newSize = numFeatureMaps * newRowSize;
            buffer = new Float32Array(newSize);
            for (let f = 0; f < numFeatureMaps; ++f) {
              const oldOffset = f * oldRowSize;
              const newOffset = f * newRowSize + f % group * oldRowSize;
              buffer.set(tensor.numberData.subarray(oldOffset, oldOffset + oldRowSize), newOffset);
            }
          }
          return this.createTextureData(adjustedLayout, tensor.type, buffer, tensor, Encoder.Usage.UploadOnly);
        }
      }

      if (textureType === TextureType.packed) {
        const unpackedTextureLayout =
            createTextureLayoutFromShape(this.session.layoutStrategy, tensor.dims, 1, [], {reverseWH: true});
        const unpackedTextureData = this.createTextureData(
            unpackedTextureLayout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
        td = this.pack(unpackedTextureData);
      } else {
        td = this.createTextureData(layout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
      }
    }
    return td;
  }

  // /**
  //  * Create a TextureData object from the given data type and texture layout.
  //  * Usage = Encoder.Usage.Default.
  //  * @param dataType the tensor data type
  //  */
  // private createTextureDataFromLayout(layout: TextureLayout, dataType: Tensor.DataType): TextureData {
  //   return this.createTextureData(layout, dataType);
  // }

  /**
   * Create a TextureData object using the given data and bind to the given tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * NOTE: this function is a hack for Conv implementation. should remove this function, after rewriting Conv
   * implementation by Graph.Transformer
   * @param dataType the tensor data type
   * @param data the actual data to upload
   * @param tensor the tensor to bind. tensor's data is ignored.
   */
  createTextureDataFromLayoutBindTensor(
      layout: TextureLayout, dataType: Tensor.DataType, data: Tensor.NumberType, tensor: Tensor): TextureData {
    return this.createTextureData(layout, dataType, data, tensor, Encoder.Usage.UploadOnly);
  }

  private createTextureData(
      layout: TextureLayout, dataType: Tensor.DataType, data?: Tensor.NumberType, tensor?: Tensor,
      usage?: Encoder.Usage): TextureData {
    Logger.verbose('InferenceHandler', `Creating TextureData: layout:[${JSON.stringify(layout)}]`);
    const texture = this.session.textureManager.createTextureFromLayout(dataType, layout, data, usage);
    return this.createTextureDataFromTexture(layout, dataType, texture, tensor);
  }

  reshapeUnpacked(input: Tensor, reshapedDims: readonly number[]): Tensor {
    const inputTD = this.getOrCreateTextureData(input, TextureType.unpacked);
    if (!inputTD) {
      throw new Error('texture data of input does not exists');
    }
    const newTextureLayout: TextureLayout = {
      channels: inputTD.channels,
      height: inputTD.height,
      width: inputTD.width,
      // handle reshaping into scalar Tensors
      shape: reshapedDims.length !== 0 ? reshapedDims : [1],
      strides: ShapeUtil.computeStrides(reshapedDims),
      unpackedShape: reshapedDims,
    };
    const newTextureData = this.createTextureDataFromTexture(newTextureLayout, input.type, inputTD.texture);
    return newTextureData.tensor;
  }

  // /**
  //  * Create a TextureData object, using the given texture.
  //  * This function does not create new texture. Usually used in scenarios using texture sharing. (eg. Reshape)
  //  * @param dataType the tensor data type
  //  * @param texture the WebGLTexture object to share
  //  * @param tensorId the tensor ID of the shared tensor data
  //  */
  // createSharedTextureData(  // TODO: make private
  //     layout: TextureLayout, dataType: Tensor.DataType, texture: WebGLTexture, tensorId?: Tensor.Id): TextureData {
  //   return this.createTextureDataFromTexture(layout, dataType, texture, undefined, tensorId);
  // }

  private createTextureDataFromTexture(
      layout: TextureLayout, dataType: Tensor.DataType, texture: WebGLTexture, tensor?: Tensor, tensorId?: Tensor.Id) {
    const textureData: TextureData = {
      ...layout,
      tensor: tensor ||
          new Tensor(
                  layout.unpackedShape, dataType, (_id: Tensor.Id) => this.readTexture(textureData), undefined,
                  undefined, tensorId),
      texture
    };
    this.setTextureData(textureData.tensor.dataId, textureData, layout.isPacked);
    return textureData;
  }

  private getTextureData(tensorId: Tensor.Id, isPacked = false): TextureData|undefined {
    return this.session.isInitializer(tensorId) ?
        this.session.getTextureData(tensorId, isPacked) :
        isPacked ? this.packedTextureDataCache.get(tensorId) : this.unpackedTextureDataCache.get(tensorId);
  }
  setTextureData(tensorId: Tensor.Id, td: TextureData, isPacked = false): void {
    if (this.session.isInitializer(tensorId)) {
      this.session.setTextureData(tensorId, td, isPacked);
    } else {
      (isPacked ? this.packedTextureDataCache : this.unpackedTextureDataCache).set(tensorId, td);
    }
  }
  isTextureLayoutCached(tensor: Tensor, isPacked = false): boolean {
    return !!this.getTextureData(tensor.dataId, isPacked);
  }

  dispose(): void {
    this.session.textureManager.clearActiveTextures();
    this.packedTextureDataCache.forEach(td => this.session.textureManager.releaseTexture(td));
    this.packedTextureDataCache = new Map();
    this.unpackedTextureDataCache.forEach(td => this.session.textureManager.releaseTexture(td));
    this.unpackedTextureDataCache = new Map();
  }

  readTexture(textureData: TextureData): Tensor.NumberType {
    if (textureData.isPacked) {
      return this.readTexture(this.unpack(textureData));
    }
    if (!this.session.backend.glContext.isFloat32DownloadSupported) {
      return this.session.textureManager.readUint8TextureAsFloat(encodeAsUint8(this, textureData));
    }
    return this.session.textureManager.readTexture(textureData, textureData.tensor.type, textureData.channels);
  }

  pack(input: TextureData): TextureData {
    const outputTextureData = this.executeProgram(
        createPackProgramInfo(this, input.tensor), [input.tensor]);  // TODO: fix after changes done for pack/unpack
    return outputTextureData;
  }

  unpack(input: TextureData): TextureData {
    const outputTextureData = this.executeProgram(
        createUnpackProgramInfo(this, input.tensor), [input.tensor]);  // TODO: fix after changes done for pack/unpack
    return outputTextureData;
  }
}
