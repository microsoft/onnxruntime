// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Sum implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(_attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length === 0) {
      return false;
    }

    const length = inputs[0].dims.length;
    for (let i = 1; i < inputs.length; i++) {
      if (length !== inputs[i].dims.length) {
        return false;
      }

      for (let j = 0; j < length; j++) {
        if (inputs[0].dims[j] !== inputs[i].dims[j]) {
          return false;
        }
      }
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }
    for (let i = 1; i < inputs.length; i++) {
      if (inputs[0].type !== inputs[i].type) {
        return false;
      }
    }

    return true;
  }
}
