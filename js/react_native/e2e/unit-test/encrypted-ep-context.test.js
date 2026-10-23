// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { gcm } from '@noble/ciphers/aes';
import { runEncryptedEpContextWorkflow } from '../src/EncryptedEpContextWorkflow';
import RNFS from 'react-native-fs';
import { InferenceSession } from 'onnxruntime-react-native';
import { Platform } from 'react-native';

jest.mock('react-native', () => ({ Platform: { OS: 'ios' } }), { virtual: true });
jest.mock('onnxruntime-react-native', () => ({ InferenceSession: { create: jest.fn() }, Tensor: jest.fn() }), {
  virtual: true,
});
jest.mock(
  'react-native-fs',
  () => ({
    DocumentDirectoryPath: '/test',
    MainBundlePath: '/bundle',
    exists: jest.fn().mockResolvedValue(true),
    readFile: jest.fn(),
    readFileAssets: jest.fn(),
    copyFileAssets: jest.fn().mockResolvedValue(undefined),
    mkdir: jest.fn().mockResolvedValue(undefined),
    unlink: jest.fn().mockResolvedValue(undefined),
  }),
  { virtual: true },
);

beforeEach(() => {
  Platform.OS = 'ios';
  RNFS.readFileAssets.mockReset();
  RNFS.copyFileAssets.mockClear();
  RNFS.unlink.mockClear();
  InferenceSession.create.mockReset();
  InferenceSession.create.mockResolvedValue({ release: jest.fn().mockResolvedValue(undefined) });
});

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

test('the runner initializes the binding before looking up the native compilation fixture', async () => {
  RNFS.readFile.mockResolvedValue(JSON.stringify({ plugin: '/test/plugin', sourceModel: '/test/model' }));
  const release = jest.fn().mockResolvedValue(undefined);
  const compile = jest.fn(() => {
    throw new Error('Native compilation fixture reached');
  });
  InferenceSession.create.mockImplementationOnce(async () => {
    globalThis.OrtApi = {
      __testCompileEpContextModel: compile,
      __testUnregisterEpContextPlugin: jest.fn(),
    };
    return { release };
  });
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('Native compilation fixture reached');
  expect(InferenceSession.create).toHaveBeenCalledWith('/test/model');
  expect(release).toHaveBeenCalledTimes(1);
  expect(compile).toHaveBeenCalledTimes(1);
});

test('the runner reads CI-bundled configuration when no document configuration is provisioned', async () => {
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFile.mockResolvedValue(JSON.stringify({ plugin: 'example_plugin_ep.dylib', sourceModel: 'model.onnx' }));
  globalThis.OrtApi = {};
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('ORT_RN_TEST_EP_CONTEXT=1');
  expect(RNFS.readFile).toHaveBeenLastCalledWith('/bundle/ort-encryption/config.json', 'utf8');
});

test('Android passes APK model and packaged library paths to the compile hook', async () => {
  Platform.OS = 'android';
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFileAssets.mockResolvedValue(
    JSON.stringify({ plugin: 'libexample_plugin_ep.so', sourceModel: 'model.onnx' }),
  );
  const compile = jest.fn(() => {
    throw new Error('Android compilation fixture reached');
  });
  globalThis.OrtApi = {
    __testCompileEpContextModel: compile,
    __testUnregisterEpContextPlugin: jest.fn(),
  };
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('Android compilation fixture reached');
  expect(RNFS.readFileAssets).toHaveBeenCalledWith('ort-encryption/config.json', 'utf8');
  const directory = compile.mock.calls[0][3];
  expect(RNFS.copyFileAssets).toHaveBeenCalledWith('ort-encryption/model.onnx', `${directory}/source.onnx`);
  expect(InferenceSession.create).toHaveBeenCalledWith(`${directory}/source.onnx`);
  expect(compile).toHaveBeenCalledWith(
    'libexample_plugin_ep.so',
    'rn_encryption_test',
    `${directory}/source.onnx`,
    expect.stringMatching(/^\/test\/ort-encryption-\d+$/),
  );
  expect(RNFS.unlink).toHaveBeenCalledWith(directory);
});

test('Android cleans its copied source directory if binding warmup fails', async () => {
  Platform.OS = 'android';
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFileAssets.mockResolvedValue(
    JSON.stringify({ plugin: 'libexample_plugin_ep.so', sourceModel: 'model.onnx' }),
  );
  InferenceSession.create.mockRejectedValueOnce(new Error('Warmup failed'));
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('Warmup failed');
  const source = RNFS.copyFileAssets.mock.calls[0][1];
  expect(RNFS.unlink).toHaveBeenCalledWith(source.substring(0, source.lastIndexOf('/')));
});

test('Android removes the source after compilation and cleans up even if removal fails', async () => {
  Platform.OS = 'android';
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFileAssets.mockResolvedValue(
    JSON.stringify({ plugin: 'libexample_plugin_ep.so', sourceModel: 'model.onnx' }),
  );
  const compile = jest.fn(() => ({}));
  const unregister = jest.fn();
  globalThis.OrtApi = {
    __testCompileEpContextModel: compile,
    __testUnregisterEpContextPlugin: unregister,
  };
  RNFS.unlink.mockRejectedValueOnce(new Error('Source removal failed'));
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('Source removal failed');
  const directory = compile.mock.calls[0][3];
  expect(RNFS.unlink).toHaveBeenNthCalledWith(1, `${directory}/source.onnx`);
  expect(unregister).toHaveBeenCalledWith('rn_encryption_test');
  expect(RNFS.unlink).toHaveBeenNthCalledWith(2, directory);
});

test('Android missing APK assets fail rather than skip the encrypted workflow', async () => {
  Platform.OS = 'android';
  RNFS.exists.mockResolvedValue(false);
  RNFS.readFileAssets.mockRejectedValue(new Error('Missing encryption fixture asset'));
  await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('Missing encryption fixture asset');
  expect(InferenceSession.create).not.toHaveBeenCalled();
});

test.each(['ios', 'android'])('%s bundled fixture rejects special paths in both fields', async (platform) => {
  Platform.OS = platform;
  RNFS.exists.mockResolvedValue(false);
  for (const path of ['../plugin', '.', '..']) {
    for (const config of [
      { plugin: path, sourceModel: 'model.onnx' },
      { plugin: 'example_plugin_ep.dylib', sourceModel: path },
    ]) {
      RNFS.readFile.mockResolvedValue(JSON.stringify(config));
      RNFS.readFileAssets.mockResolvedValue(JSON.stringify(config));
      await expect(runEncryptedEpContextWorkflow()).rejects.toThrow('must be filenames');
    }
  }
  expect(RNFS.copyFileAssets).not.toHaveBeenCalled();
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
