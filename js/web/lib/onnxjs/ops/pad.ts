// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Pad implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.mode = attributes.getString('mode', 'constant');
    this.value = attributes.getFloat('value', 0.0);
    this.pads = attributes.getInts('pads');
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }

    return true;
  }

  protected mode: string;
  protected pads: number[];
  protected value: number;
}
