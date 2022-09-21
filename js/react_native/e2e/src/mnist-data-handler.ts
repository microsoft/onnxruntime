// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {NativeModules} from 'react-native';

export interface MNISTInput {
  [name: string]: {
    dims: number[]; type: string; data: string;  // encoded tensor data
  };
}

export interface MNISTOutput {
  [name: string]: {
    data: string;  // encoded tensor data
  };
}

export interface MNISTResult {
  result: string;
}

type MNISTType = {
  getLocalModelPath(): Promise<string>; getImagePath(): Promise<string>; preprocess(uri: string): Promise<MNISTInput>;
  postprocess(result: MNISTOutput): Promise<MNISTResult>;
};

const MNIST = NativeModules.MNISTDataHandler;

export default MNIST as MNISTType;
