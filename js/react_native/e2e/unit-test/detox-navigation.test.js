// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';
import * as React from 'react';
import { Button } from 'react-native';
import renderer, { act } from 'react-test-renderer';
import App from '../src/App';

jest.mock('../src/MNISTTest', () => 'MNISTTest');
jest.mock('../src/BasicTypesTest', () => 'BasicTypesTest');
jest.mock('../src/EPContextDataReadTest', () => 'EPContextDataReadTest');

test('EPContext navigation identifier belongs to exactly one native button', () => {
  let tree;
  act(() => {
    tree = renderer.create(React.createElement(App));
  });
  const nativeTargets = tree.root.findAll(
    (node) => typeof node.type === 'string' && node.props.testID === 'ep-context-data-read-test-button',
  );
  expect(nativeTargets).toHaveLength(1);
  const button = tree.root.find(
    (node) => node.type === Button && node.props.testID === 'ep-context-data-read-test-button',
  );
  act(() => {
    button.props.onPress();
  });
  expect(tree.root.findByType('EPContextDataReadTest')).toBeDefined();
  act(() => {
    tree.unmount();
  });
});

test.each(['android', 'ios'])('EPContext navigation taps the unique button on %s', async (platform) => {
  const scenarios = new Map();
  let initialize;
  const element = jest.fn((matcher) => {
    if (platform === 'ios' && matcher.label === 'ep-context-data-read-test-button') {
      throw new Error('Multiple native elements share the EPContext accessibility label');
    }
    return { tap: jest.fn().mockResolvedValue(undefined) };
  });
  const waitFor = jest.fn(() => ({
    toHaveText: jest.fn(() => ({ withTimeout: jest.fn().mockResolvedValue(undefined) })),
  }));
  runInNewContext(readFileSync(join(__dirname, '../test/OnnxruntimeModuleExample.test.js'), 'utf8'), {
    describe: (_name, register) => register(),
    beforeAll: (callback) => {
      initialize = callback;
    },
    beforeEach: jest.fn(),
    it: (name, callback) => scenarios.set(name, callback),
    device: { getPlatform: () => platform, launchApp: jest.fn().mockResolvedValue(undefined) },
    element,
    by: { id: (id) => ({ id }), label: (label) => ({ label }), text: (text) => ({ text }) },
    waitFor,
    setTimeout: (callback) => callback(),
  });

  await initialize();
  await scenarios.get('EPContext data read test should run successfully')();

  expect(element).toHaveBeenNthCalledWith(1, { id: 'ep-context-data-read-test-button' });
  expect(element.mock.results[0].value.tap).toHaveBeenCalledTimes(1);
  expect(waitFor).toHaveBeenCalledWith({ tap: expect.any(Function) });
  expect(waitFor.mock.results[0].value.toHaveText).toHaveBeenCalledWith('14/14 checks passed');
});
