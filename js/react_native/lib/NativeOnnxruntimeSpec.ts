// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line import/no-internal-modules
import type { TurboModule } from 'react-native/Libraries/TurboModule/RCTExport.js';
import { TurboModuleRegistry } from 'react-native';

// NOTE: Currently we can't use types import from another files
// ref: https://github.com/facebook/react-native/issues/36431
export interface Spec extends TurboModule {
  loadModel(modelPath: string, options: object): Promise<object>;
  loadModelFromBlob?(blob: object, options: object): Promise<object>;
  dispose(key: string): Promise<void>;
  run(key: string, feeds: object, fetches: object, options: object): Promise<object>;
}

export default TurboModuleRegistry.get<Spec>('Onnxruntime');
