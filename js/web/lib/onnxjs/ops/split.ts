// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Split implements Operator {
  constructor(protected numOutputs?: number) {}

  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.axis = attributes.getInt('axis', 0);
    this.split = attributes.getInts('split', []);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'int8' && inputs[0].type !== 'uint8' && inputs[0].type !== 'int16' &&
        inputs[0].type !== 'uint16' && inputs[0].type !== 'int32' && inputs[0].type !== 'uint32' &&
        inputs[0].type !== 'float32' && inputs[0].type !== 'float64' && inputs[0].type !== 'bool') {
      return false;
    }

    return true;
  }

  protected split: number[];
  protected axis: number;
}
