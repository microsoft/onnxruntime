// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

describe('OnnxruntimeModuleExample', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.launchApp({ newInstance: true});
  });

  it('TEST should have accessibilityLabel', async () => {
    await element(by.label('output'));
  });

  it('TEST output inference result value', async () => {
    await expect(element(by.text('Result: 3')));
  });
});