// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {creatPackProgramInfo} from './ops/pack';

import {WebGLUint8Encode} from './ops/uint8-encode';
import {creatUnpackProgramInfo} from './ops/unpack';
import {WebGLSessionHandler} from './session-handler';
import {Encoder} from './texture-data-encoder';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {Artifact, ProgramInfo, TextureData, TextureLayout, TextureType} from './types';
import {getPackedShape} from './utils';

const getProgramInfoUniqueKey = (programInfo: ProgramInfo, InputTextureDatas: TextureData[]): string => {
  // TODO
};

export class WebGLInferenceHandler implements InferenceHandler {
  private packedTextureDataCache: Map<Tensor.Id, TextureData>;
  private unpackedTextureDataCache: Map<Tensor.Id, TextureData>;
  private pack2unpackMap: Map<Tensor.Id, Tensor.Id>;
  private unpack2packMap: Map<Tensor.Id, Tensor.Id>;
  constructor(public session: WebGLSessionHandler) {
    this.packedTextureDataCache = new Map();
    this.unpackedTextureDataCache = new Map();

    this.pack2unpackMap = new Map();
    this.unpack2packMap = new Map();
  }

  /**
   * @returns [width, height]
   */
  calculateTextureWidthAndHeight(shape: readonly number[], textureType: TextureType): [number, number] {
    const layout = this.createTextureLayoutFromTextureType(shape, textureType);
    return [layout.width, layout.height];
  }

  /*private checkAndUpdateTextureForm(programInfo: ProgramInfo, inputs: Tensor[]) {
    // pack/unpack inputs
    for (let i = 0; i < inputs.length; ++i) {
      const input = inputs[i];
      if (input.isPacked && programInfo.inputTypes[i] !== TextureType.packed) {
        inputs[i] = this.unpack(input);
      } else if (!input.isPacked && artifact.programInfo.inputTypes[i] === TextureType.packed) {
        inputs[i] = this.pack(input);
      }
    }
  }*/

  executeProgram(programInfo: ProgramInfo, inputs: Tensor[]) :TextureData{
    // create texture info for input
    const inputTextureDatas = inputs.map((tensor, i) => {
      const textureType = programInfo.inputTypes[i];
      let td = this.getTextureData(tensor.dataId, textureType === TextureType.packed);
      if (!td) {
        const layout = this.createTextureLayoutFromTextureType(tensor.dims, textureType);

        if (textureType === TextureType.packed) {
          const unpackedTextureLayout = this.getOrCreateTextureLayout(tensor, 1, false, [], true);
          const unpackedTextureData = this.createTextureData(
              unpackedTextureLayout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
          td = this.pack(unpackedTextureData);
        } else {
          td = this.createTextureData(layout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
        }
      }
      return td;
    });

    // create texture info for output
    const outputTextureLayout =
        this.createTextureLayoutFromTextureType(programInfo.output.dims, programInfo.output.textureType);
    const outputTextureData = this.createTextureData(outputTextureLayout, programInfo.output.type);

    const key = getProgramInfoUniqueKey(programInfo, inputTextureDatas);
    let artifact = this.session.programManager.getArtifact(key);
    if (!artifact) {
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(key, artifact);
    }

    this.runProgram(artifact, inputTextureDatas, outputTextureData);
    return outputTextureData;
  }

  run(programInfo: ProgramInfo, inputs: Tensor[]): Tensor {
    const outputTextureData = this.executeProgram(programInfo, inputs);
    //this.checkAndUpdateTextureForm(programInfo, inputs);
    // create texture info for input
    // const inputTextureDatas = inputs.map((tensor, i) => {
    //   const textureType = programInfo.inputTypes[i];
    //   let td = this.getTextureData(tensor.dataId, textureType === TextureType.packed);
    //   if (!td) {
    //     const layout = this.createTextureLayoutFromTextureType(tensor.dims, textureType);

    //     if (textureType === TextureType.packed) {
    //       const unpackedTextureLayout = this.getOrCreateTextureLayout(tensor, 1, false, [], true);
    //       const unpackedTextureData = this.createTextureData(
    //           unpackedTextureLayout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
    //       td = this.pack(unpackedTextureData);
    //     } else {
    //       td = this.createTextureData(layout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
    //     }
    //   }
    //   return td;
    // });

    // // create texture info for output
    // const outputTextureLayout =
    //     this.createTextureLayoutFromTextureType(programInfo.output.dims, programInfo.output.textureType);
    // const outputTextureData = this.createTextureData(outputTextureLayout, programInfo.output.type);

    // const key = getProgramInfoUniqueKey(programInfo, inputTextureDatas);
    // let artifact = this.session.programManager.getArtifact(key);
    // if (!artifact) {
    //   artifact = this.session.programManager.build(programInfo);
    //   this.session.programManager.setArtifact(key, artifact);
    // }

    // this.runProgram(artifact, inputTextureDatas, outputTextureData);
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
  private getOrCreateTextureData(tensor: Tensor, layout?: TextureLayout, isPacked = false) {
    let td = this.getTextureData(tensor.dataId, isPacked);
    if (!td) {
      Logger.verbose('InferenceHandler', `Creating new TextureData for dims: [${tensor.dims}]`);
      if (!layout) {
        layout = this.createTextureLayoutFromShape(tensor.dims.slice());
      }
      // if we don't find the texture data with specific pack mode in the cache, try with the different
      // pack mode to see if the tensor is cached using that pack mode. If succeed, we can return this
      // tensor data and later apply a pack/unpack op on this texture, no need to create a new one here.
      td = this.getTextureData(tensor.dataId, !isPacked);
      if (!td) {
        if (isPacked) {
          const unpackedTextureLayout = this.getOrCreateTextureLayout(tensor, 1, false, [], true);
          const unpackedTextureData = this.createTextureData(
              unpackedTextureLayout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
          td = this.pack(unpackedTextureData);
        } else {
          td = this.createTextureData(layout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
        }
      }
    } else {
      Logger.verbose('InferenceHandler', `Retrieving TextureData from cache: [${tensor.dims}]`);
    }
    return td;
  }

  /**
   * Create a TextureData object from the given data type and texture layout.
   * Usage = Encoder.Usage.Default.
   * @param dataType the tensor data type
   */
  private createTextureDataFromLayout(layout: TextureLayout, dataType: Tensor.DataType): TextureData {
    return this.createTextureData(layout, dataType);
  }

  /**
   * Create a TextureData object using the given data and bind to the given tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * NOTE: this function is a hack for Conv implementation. should remove this function, after rewriting Conv
   * implementation by Graph.Transformer
   * @param dataType the tensor data type
   * @param data the actual data to upload
   * @param tensor the tensor to bind. tensor's data is ignored.
   */
  private createTextureDataFromLayoutBindTensor(
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

  /**
   * Create a TextureData object, using the given texture.
   * This function does not create new texture. Usually used in scenarios using texture sharing. (eg. Reshape)
   * @param dataType the tensor data type
   * @param texture the WebGLTexture object to share
   * @param tensorId the tensor ID of the shared tensor data
   */
  createSharedTextureData(  // TODO: make private
      layout: TextureLayout, dataType: Tensor.DataType, texture: WebGLTexture, tensorId?: Tensor.Id): TextureData {
    return this.createTextureDataFromTexture(layout, dataType, texture, undefined, tensorId);
  }

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
  private setTextureData(tensorId: Tensor.Id, td: TextureData, isPacked = false): void {
    if (this.session.isInitializer(tensorId)) {
      this.session.setTextureData(tensorId, td, isPacked);
    } else {
      (isPacked ? this.packedTextureDataCache : this.unpackedTextureDataCache).set(tensorId, td);
    }
  }
  isTextureLayoutCached(tensor: Tensor, isPacked = false): boolean {
    return !!this.getTextureData(tensor.dataId, isPacked);
  }
  /**
   * Create a TextureLayout object from a tensor. If a related texture data is found, returns the cached texture layout.
   */
  private getOrCreateTextureLayout(
      tensor: Tensor, channels: 1|4 = 1, isPacked = false, unpackedShape?: readonly number[],
      reverseWH = false): TextureLayout {
    const td = this.getTextureData(tensor.dataId, isPacked);
    if (td) {
      return td;
    }
    return this.createTextureLayoutFromShape(
        channels === 1 || isPacked ? tensor.dims : getPackedShape(tensor.dims), channels, unpackedShape,
        isPacked || reverseWH ? {isPacked, reverseWH} : undefined);
  }

  private createTextureLayoutFromTextureType(  // TODO: rename this function and (possibly?) move out of this class
      shape: readonly number[], textureType: TextureType): TextureLayout {
    return this.createTextureLayoutFromShape(
        shape, textureType === TextureType.unpacked ? 1 : 4, [],
        textureType === TextureType.packed ? {isPacked: true, reverseWH: true} : undefined);
  }

  /**
   * Create a TextureLayout object from shape.
   */
  private createTextureLayoutFromShape(
      shape: readonly number[], channels: 1|4 = 1, unpackedShape?: readonly number[],
      prefs?: WidthHeightPrefs): TextureLayout {
    const isPacked = !!(prefs && prefs.isPacked);
    const [texWidth, texHeight] =
        this.session.layoutStrategy.computeTextureWH(isPacked ? unpackedShape || shape : shape, prefs);
    let [width, height] = [texWidth, texHeight];
    if (prefs && prefs.reverseWH) {
      width = texHeight;
      height = texWidth;
    }
    const rank = shape.length;
    let inferredDims = shape.slice(0);
    if (rank === 0) {
      inferredDims = [1];
    }
    if (channels === 1) {
      // unpackedShape will take `shape` and not `inferredDims` so as to create a scalar Tensor if need be
      unpackedShape = shape;
    } else if (isPacked) {
      if (channels !== 4) {
        throw new Error('a packed texture must be 4-channel');
      }
      unpackedShape = shape;
      if (rank > 0) {
        inferredDims[rank - 1] = Math.ceil(inferredDims[rank - 1] / 2);
      }
      if (rank > 1) {
        inferredDims[rank - 2] = Math.ceil(inferredDims[rank - 2] / 2);
      }
    } else if (!unpackedShape) {
      throw new Error('Unpacked shape is needed when using channels > 1');
    }
    return {
      width,
      height,
      channels,
      isPacked,
      shape: inferredDims,
      strides: ShapeUtil.computeStrides(inferredDims),
      unpackedShape,
      reversedWH: (prefs && prefs.reverseWH)
    };
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
      const op = new WebGLUint8Encode();
      const uint8TD = op.runInternal(this, textureData);
      return this.session.textureManager.readUint8TextureAsFloat(uint8TD);
    }
    return this.session.textureManager.readTexture(textureData, textureData.tensor.type, textureData.channels);
  }

  pack(input: TextureData): TextureData {

    //const runData = op.createRunData(this, artifact.programInfo, [input.tensor]);
    const outputTextureData = this.executeProgram(creatPackProgramInfo(this, input.tensor), inputs);  // TODO: fix after changes done for pack/unpack
    return outputTextureData;
  }

  unpack(input: TextureData): TextureData {
    const outputTextureData = this.executeProgram(creatUnpackProgramInfo(this, input.tensor), inputs);  // TODO: fix after changes done for pack/unpack
    return outputTextureData;
  }
}
