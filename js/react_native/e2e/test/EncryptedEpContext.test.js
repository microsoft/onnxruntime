// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const describeEncryptedWorkflow = process.env.ORT_RN_ENCRYPTION_E2E === '1' ? describe : describe.skip;

describeEncryptedWorkflow('Encrypted compiled EPContext inference (real mobile binding)', () => {
  it('compiles, persists ciphertext, decrypts, runs inference, and rejects invalid context', async () => {
    await device.launchApp({ newInstance: true });
    if (device.getPlatform() === 'ios') {
      await element(by.text('EPContext Data Read Test')).tap();
      await element(by.text('Run Encrypted EPContext Workflow')).tap();
    } else {
      await element(by.label('ep-context-data-read-test-button')).tap();
      await element(by.label('run-encrypted-ep-context')).tap();
    }
    await waitFor(element(by.label('encrypted-ep-context-result')))
      .toHaveText('Encrypted compiled model/context: two inference cases and all negative controls passed')
      .withTimeout(120000);
  });
});
