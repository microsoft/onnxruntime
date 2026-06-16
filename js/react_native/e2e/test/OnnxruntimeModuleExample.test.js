// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* Initial Simple Detox Test Setup. Can potentially add more unit tests. */

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

describe('OnnxruntimeModuleExample', () => {
  let platform;

  beforeAll(async () => {
    platform = device.getPlatform();
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('MNIST test inference result should be correct', async () => {
    // Tap MNIST test button
    if (platform === 'android') {
      await element(by.label('mnist-test-button')).tap();
    } else {
      await element(by.text('MNIST Test')).tap();
    }

    await delay(500);

    // Check the inference result
    if (platform === 'ios') {
      await expect(element(by.label('output')).atIndex(1)).toHaveText('Result: 3');
    }
    if (platform === 'android') {
      await expect(element(by.label('output'))).toHaveText('Result: 3');
    }
  });

  it('Basic Types test should run successfully', async () => {
    // Tap Basic Types test button
    if (platform === 'android') {
      await element(by.label('basic-types-test-button')).tap();
    } else {
      await element(by.text('Basic Types Test')).tap();
    }

    await delay(500);

    // Run the tests
    if (platform === 'android') {
      await element(by.label('run-tests-button')).tap();
    } else {
      await element(by.text('Run All Tests')).tap();
    }
    await delay(500);

    // Check have no error
    await expect(element(by.label('statusError'))).not.toBeVisible();
  });
});
