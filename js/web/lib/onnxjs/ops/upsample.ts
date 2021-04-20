// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Upsample implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.mode = attributes.getString('mode', 'nearest');
    this.scales = attributes.getFloats('scales');

    if (this.mode !== 'nearest' && this.mode !== 'linear') {
      throw new Error(`unrecognized mode: ${this.mode}`);
    }

    if (this.mode === 'linear' && this.scales.length !== 2 && this.scales.length !== 4) {
      throw new Error('only support 2-D or 4-D upsampling for linear mode');
    }

    this.roi = new Array<number>(this.scales.length * 2).fill(0);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    if (inputs[0].dims.length !== this.scales.length) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type === 'string') {
      return false;
    }

    return true;
  }

  protected mode: string;
  protected scales: number[];
  protected roi: number[];
}

export abstract class UpsampleV9 implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.mode = attributes.getString('mode', 'nearest');

    if (this.mode !== 'nearest' && this.mode !== 'linear') {
      throw new Error(`unrecognized mode: ${this.mode}`);
    }
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 2) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type === 'string') {
      return false;
    }

    return true;
  }

  protected mode: string;
}
