// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend, env} from 'onnxruntime-common';
import {Platform} from 'react-native';
import {onnxruntimeBackend} from './backend';
import {version} from './version';

registerBackend('cpu', onnxruntimeBackend, 1);
registerBackend('xnnpack', onnxruntimeBackend, 1);
if (Platform.OS === 'android') {
  registerBackend('nnapi', onnxruntimeBackend, 1);
} else if (Platform.OS === 'ios') {
  registerBackend('coreml', onnxruntimeBackend, 1);
}

Object.defineProperty(env.versions, 'react-native', {value: version, enumerable: true});
