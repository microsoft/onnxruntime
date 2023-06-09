// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {Platform} from 'react-native';
import {registerBackend} from 'onnxruntime-common';
import {onnxruntimeBackend} from './backend';

registerBackend('cpu', onnxruntimeBackend, 1);
registerBackend('xnnpack', onnxruntimeBackend, 1);
if (Platform.OS === 'android') {
  registerBackend('nnapi', onnxruntimeBackend, 1);
} else if (Platform.OS === 'ios') {
  registerBackend('coreml', onnxruntimeBackend, 1);
}
