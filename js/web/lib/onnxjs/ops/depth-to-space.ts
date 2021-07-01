// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class DepthToSpace implements Operator {
  constructor() {}

  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    // processing node attributes
    this.blocksize = attributes.getInt('blocksize');
    if (this.blocksize < 1) {
      throw new Error(`blocksize must be >= 1, but got : ${this.blocksize} for DepthToSpace`);
    }
    this.blocksizeSqr = this.blocksize * this.blocksize;
    this.mode = attributes.getString('mode', 'DCR');
    if (DepthToSpace.supportedModes.indexOf(this.mode) < 0) {
      throw new Error(`unrecognized mode: ${this.mode} for DepthToSpace`);
    }
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    const inputType = inputs[0].type;
    const inputDimensionality = inputs[0].dims.length;

    // Input has to be a 4-D tensor
    // TODO: Support string depth-to-space.
    if (inputType === 'string' || inputDimensionality !== 4) {
      return false;
    }

    return true;
  }

  protected mode: string;
  protected blocksize: number;
  protected blocksizeSqr: number;

  private static readonly supportedModes = ['DCR', 'CRD'];
}