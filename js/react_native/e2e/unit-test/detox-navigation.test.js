// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';

test.each(['android', 'ios'])('EPContext navigation taps the accessible button on %s', async (platform) => {
  const scenarios = new Map();
  let initialize;
  const element = jest.fn(() => ({ tap: jest.fn().mockResolvedValue(undefined) }));
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
    by: { label: (label) => ({ label }), text: (text) => ({ text }) },
    waitFor,
    setTimeout: (callback) => callback(),
  });

  await initialize();
  await scenarios.get('EPContext data read test should run successfully')();

  expect(element).toHaveBeenNthCalledWith(1, { label: 'ep-context-data-read-test-button' });
  expect(element.mock.results[0].value.tap).toHaveBeenCalledTimes(1);
  expect(waitFor).toHaveBeenCalledWith({ tap: expect.any(Function) });
  expect(waitFor.mock.results[0].value.toHaveText).toHaveBeenCalledWith('14/14 checks passed');
});
