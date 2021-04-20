// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {WebGLPack} from './ops/pack';

import {WebGLUint8Encode} from './ops/uint8-encode';
import {WebGLUnpack} from './ops/unpack';
import {WebGLSessionHandler} from './session-handler';
import {Encoder} from './texture-data-encoder';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {Artifact, RunData, TextureData, TextureLayout, WebGLOperator} from './types';
import {getPackedShape} from './utils';

export class WebGLInferenceHandler implements InferenceHandler {
  private textureDataCache: Map<Tensor.Id, TextureData>;
  constructor(public session: WebGLSessionHandler) {
    this.textureDataCache = new Map();
  }

  run(op: WebGLOperator, inputs: Tensor[]): Tensor[] {
    let artifact = this.session.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(this, inputs);
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(this, artifact.programInfo, inputs);
    this.runProgram(artifact, runData);
    return [runData.outputTextureData.tensor];
  }

  runProgram(artifact: Artifact, runData: RunData) {
    // pack/unpack inputs
    runData.inputTextureDatas.forEach(input => {
      if (input.isPacked && !artifact.programInfo.expectPackedInputs) {
        // unpack this input
        const unpacked = this.unpack(input);
        input.height = unpacked.height;
        input.isPacked = unpacked.isPacked;
        input.texture = unpacked.texture;
        input.width = unpacked.width;

      } else if (!input.isPacked && artifact.programInfo.expectPackedInputs) {
        // pack this input
        const packed = this.pack(input);
        input.height = packed.height;
        input.isPacked = packed.isPacked;
        input.texture = packed.texture;
        input.width = packed.width;
      }
    });

    // output should match
    if (!!runData.outputTextureData.isPacked !== !!artifact.programInfo.expectPackedoutputs) {
      throw new Error('output property packed inconsistent');
    }

    this.session.programManager.run(artifact, runData);
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
  getOrCreateTextureData(tensor: Tensor, layout?: TextureLayout) {
    let td = this.getTextureData(tensor.dataId);
    if (!td) {
      Logger.verbose('InferenceHandler', `Creating new TextureData for dims: [${tensor.dims}]`);
      if (!layout) {
        layout = this.createTextureLayoutFromShape(tensor.dims.slice());
      }
      // graph inputs or initializers
      td = this.createTextureData(layout, tensor.type, tensor.numberData, tensor, Encoder.Usage.UploadOnly);
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
  createTextureDataFromLayout(layout: TextureLayout, dataType: Tensor.DataType): TextureData {
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

  /**
   * Create a TextureData object, using the given texture.
   * This function does not create new texture. Usually used in scenarios using texture sharing. (eg. Reshape)
   * @param dataType the tensor data type
   * @param texture the WebGLTexture object to share
   * @param tensorId the tensor ID of the shared tensor data
   */
  createSharedTextureData(layout: TextureLayout, dataType: Tensor.DataType, texture: WebGLTexture, tensorId: Tensor.Id):
      TextureData {
    return this.createTextureDataFromTexture(layout, dataType, texture, undefined, tensorId);
  }

  private createTextureDataFromTexture(
      layout: TextureLayout, dataType: Tensor.DataType, texture: WebGLTexture, tensor?: Tensor, tensorId?: Tensor.Id) {
    const textureData: TextureData = {
      ...layout,
      tensor: tensor ||
          new Tensor(
                  layout.unpackedShape, dataType,
                  (_id: Tensor.Id) => {
                    const data = this.readTexture(textureData);
                    if (dataType === 'float32') {
                      return Float32Array.from(data);
                    } else if (dataType === 'bool') {
                      return Uint8Array.from(data);
                    }
                    return data;
                  },
                  undefined, undefined, tensorId),
      texture
    };
    this.setTextureData(textureData.tensor.dataId, textureData);
    return textureData;
  }

  getTextureData(tensorId: Tensor.Id): TextureData|undefined {
    return this.session.isInitializer(tensorId) ? this.session.getTextureData(tensorId) :
                                                  this.textureDataCache.get(tensorId);
  }
  setTextureData(tensorId: Tensor.Id, td: TextureData): void {
    if (this.session.isInitializer(tensorId)) {
      this.session.setTextureData(tensorId, td);
    } else {
      this.textureDataCache.set(tensorId, td);
    }
  }

  /**
   * Create a TextureLayout object from a tensor. If a related texture data is found, returns the cached texture layout.
   */
  getOrCreateTextureLayout(
      tensor: Tensor, channels: 1|4 = 1, isPacked = false, unpackedShape?: readonly number[],
      reverseWH = false): TextureLayout {
    const td = this.getTextureData(tensor.dataId);
    if (td) {
      return td;
    }
    return this.createTextureLayoutFromShape(
        channels === 1 || isPacked ? tensor.dims : getPackedShape(tensor.dims), channels, unpackedShape,
        isPacked || reverseWH ? {isPacked, reverseWH} : undefined);
  }
  /**
   * Create a TextureLayout object from shape.
   */
  createTextureLayoutFromShape(
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
      unpackedShape
    };
  }

  dispose(): void {
    this.session.textureManager.clearActiveTextures();
    this.textureDataCache.forEach(td => this.session.textureManager.releaseTexture(td));
    this.textureDataCache = new Map();
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
    const key = `${input.shape}`;
    let op = this.session.packOpCache.get(key);
    if (!op) {
      op = new WebGLPack();
      this.session.packOpCache.set(key, op);
    }
    let artifact = this.session.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(this, [input.tensor]);
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(this, artifact.programInfo, [input.tensor]);
    this.runProgram(artifact, runData);
    return runData.outputTextureData;
  }

  unpack(input: TextureData): TextureData {
    const key = `${input.shape}`;
    let op = this.session.unpackOpCache.get(key);
    if (!op) {
      op = new WebGLUnpack();
      this.session.unpackOpCache.set(key, op);
    }
    let artifact = this.session.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(this, [input.tensor]);
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(this, artifact.programInfo, [input.tensor]);
    this.runProgram(artifact, runData);
    return runData.outputTextureData;
  }
}
