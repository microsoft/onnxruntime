// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { gcm } from '@noble/ciphers/aes';
import { runEncryptedEpContextWorkflow } from '../src/EncryptedEpContextWorkflow';
import RNFS from 'react-native-fs';

jest.mock('onnxruntime-react-native', () => ({ InferenceSession: {}, Tensor: jest.fn() }), { virtual: true });
jest.mock(
  'react-native-fs',
  () => ({
    DocumentDirectoryPath: '/test',
    MainBundlePath: '/bundle',
    exists: jest.fn().mockResolvedValue(true),
    readFile: jest.fn(),
  }),
  { virtual: true },
);

afterEach(() => {
  delete globalThis.OrtApi;
  RNFS.exists.mockResolvedValue(true);
});

test('the opt-in runner rejects invalid configuration rather than reporting a skipped success', async () => {
  RNFS.readFile.mockResolvedValue(JSON.stringify({ plugin: 1 }));
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('requires plugin and sourceModel paths');
});

test('the runner rejects production builds without the native compilation fixture', async () => {
  RNFS.readFile.mockResolvedValue(JSON.stringify({ plugin: '/test/plugin', sourceModel: '/test/model' }));
  globalThis.OrtApi = {};
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('ORT_RN_TEST_EP_CONTEXT=1');
});

test('the runner reads CI-bundled configuration when no document configuration is provisioned', async () => {
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFile.mockResolvedValue(JSON.stringify({ plugin: 'example_plugin_ep.dylib', sourceModel: 'model.onnx' }));
  globalThis.OrtApi = {};
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('ORT_RN_TEST_EP_CONTEXT=1');
  expect(RNFS.readFile).toHaveBeenLastCalledWith('/bundle/ort-encryption/config.json', 'utf8');
});

test.each(['../plugin', '.', '..'])('bundled fixture rejects special path %s', async (path) => {
  RNFS.exists.mockResolvedValue(false);
  for (const config of [
    { plugin: path, sourceModel: 'model.onnx' },
    { plugin: 'example_plugin_ep.dylib', sourceModel: path },
  ]) {
    RNFS.readFile.mockResolvedValue(JSON.stringify(config));
    await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('must be filenames');
  }
});

test('the mobile AES-GCM dependency round trips and rejects wrong keys, tampering, and asset names', () => {
  // This is a crypto unit test, not compiled-model or JSI inference coverage.
  const key = new Uint8Array(32);
  const nonce = new Uint8Array(12);
  const name = new Uint8Array([1, 2, 3]);
  const plaintext = new Uint8Array([4, 5, 6]);
  const ciphertext = gcm(key, nonce, name).encrypt(plaintext);
  expect(gcm(key, nonce, name).decrypt(ciphertext)).toEqual(plaintext);
  const wrongKey = key.slice();
  wrongKey[0] = 1;
  expect(() => gcm(wrongKey, nonce, name).decrypt(ciphertext)).toThrow();
  const tampered = ciphertext.slice();
  tampered[0] ^= 1;
  expect(() => gcm(key, nonce, name).decrypt(tampered)).toThrow();
  expect(() => gcm(key, nonce, new Uint8Array([9])).decrypt(ciphertext)).toThrow();
});
