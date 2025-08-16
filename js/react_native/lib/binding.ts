import { NativeModules } from 'react-native';
import type { OrtApi as OrtApiType } from './api';

export const Module = NativeModules.Onnxruntime;

if (typeof globalThis.OrtApi === 'undefined') {
  Module.install();
}

export const OrtApi = globalThis.OrtApi as OrtApiType;
