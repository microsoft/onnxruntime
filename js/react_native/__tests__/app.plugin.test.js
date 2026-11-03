jest.mock('@expo/config-plugins', () => ({
  withAppBuildGradle: jest.fn(),
  withDangerousMod: jest.fn(),
  withXcodeProject: jest.fn(),
  IOSConfig: { Paths: { getSourceRoot: jest.fn() }, XcodeUtils: { addResourceFileToGroup: jest.fn() } },
  createRunOncePlugin: jest.fn((fn) => fn),
}), { virtual: true });
jest.mock('@expo/config-plugins/build/utils/generateCode', () => ({}), { virtual: true });

const fs = require('fs');
const os = require('os');
const path = require('path');

const {
  _resolveAndValidateModels: resolveAndValidateModels,
  _warnIfMetroAssetExtsMissing: warnIfMetroAssetExtsMissing,
} = require('../app.plugin');

let tmpDir;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-plugin-test-'));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('resolveAndValidateModels', () => {
  let warnSpy;

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('resolves relative paths and returns src + filename', () => {
    const modelFile = path.join(tmpDir, 'model.onnx');
    fs.writeFileSync(modelFile, 'data');

    const result = resolveAndValidateModels(tmpDir, ['./model.onnx']);

    expect(result).toEqual([
      { src: modelFile, filename: 'model.onnx' },
    ]);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('warns and skips missing files', () => {
    const result = resolveAndValidateModels(tmpDir, ['./missing.onnx']);

    expect(result).toEqual([]);
    expect(warnSpy).toHaveBeenCalledWith(
      '[onnxruntime-react-native]',
      expect.stringContaining('Model file not found: ./missing.onnx'),
    );
  });

  it('warns on duplicate basenames', () => {
    const subDir = path.join(tmpDir, 'sub');
    fs.mkdirSync(subDir);
    fs.writeFileSync(path.join(tmpDir, 'model.onnx'), 'data1');
    fs.writeFileSync(path.join(subDir, 'model.onnx'), 'data2');

    const result = resolveAndValidateModels(tmpDir, [
      './model.onnx',
      './sub/model.onnx',
    ]);

    expect(result).toHaveLength(2);
    expect(warnSpy).toHaveBeenCalledWith(
      '[onnxruntime-react-native]',
      expect.stringContaining('same filename "model.onnx"'),
      expect.any(String),
      expect.any(String),
      expect.any(String),
    );
  });

  it('returns empty array when all paths are invalid', () => {
    const result = resolveAndValidateModels(tmpDir, [
      './a.onnx',
      './b.ort',
    ]);

    expect(result).toEqual([]);
    expect(warnSpy).toHaveBeenCalledTimes(2);
  });
});

describe('warnIfMetroAssetExtsMissing', () => {
  let warnSpy;

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('warns when metro config exists but missing extensions', () => {
    const configContent = 'module.exports = { resolver: { assetExts: ["png", "jpg"] } };';
    fs.writeFileSync(path.join(tmpDir, 'metro.config.js'), configContent);

    warnIfMetroAssetExtsMissing(tmpDir);

    expect(warnSpy).toHaveBeenCalledWith(
      '[onnxruntime-react-native]',
      expect.stringContaining('onnx, ort'),
      expect.any(String),
      expect.any(String),
    );
  });

  it('does not warn when extensions are present', () => {
    const configContent = 'module.exports = { resolver: { assetExts: ["png", "onnx", "ort"] } };';
    fs.writeFileSync(path.join(tmpDir, 'metro.config.js'), configContent);

    warnIfMetroAssetExtsMissing(tmpDir);

    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('does not warn when no metro config file exists', () => {
    warnIfMetroAssetExtsMissing(tmpDir);

    expect(warnSpy).not.toHaveBeenCalled();
  });
});
