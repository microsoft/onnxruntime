// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Concat implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.axis = attributes.getInt('axis');
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length < 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    const inputType = inputs[0].type;
    const inputDimensionality = inputs[0].dims.length;

    // TODO: Support string concat
    if (inputType === 'string') {
      return false;
    }

    for (const input of inputs) {
      // make sure types of all inputs match
      if (input.type !== inputType) {
        return false;
      }

      // make sure the dimensionality of all inputs are the same
      if (input.dims.length !== inputDimensionality) {
        return false;
      }
    }

    return true;
  }

  protected axis: number;
}
