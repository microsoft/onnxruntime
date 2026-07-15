// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const EventEmitter = require('node:events');
const path = require('path');
const { runShellCmd } = require('./utils');

const launchBrowserAndRunTests = async (logPrefix, port) => {
  const testResults = [];
  const puppeteer = require('puppeteer-core');
  let browser;
  try {
    browser = await puppeteer.launch({
      executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      browser: 'chrome',
      headless: true,
      args: ['--enable-features=SharedArrayBuffer', '--no-sandbox', '--disable-setuid-sandbox'],
    });

    for await (const flags of [
      [false, false],
      [false, true],
      [true, false],
      [true, true],
    ]) {
      const [multiThread, proxy] = flags;
      console.log(`[${logPrefix}] Running test with multi-thread: ${multiThread}, proxy: ${proxy}...`);

      const page = await browser.newPage();
      await page.goto(`http://localhost:${port}`);
      // wait for the page to load
      await page.waitForSelector('#ortstate', { visible: true });
      // if multi-thread is enabled, check the checkbox
      if (multiThread) {
        await page.locator('#cb-mt').click();
      }
      // if proxy is enabled, check the checkbox
      if (proxy) {
        await page.locator('#cb-px').click();
      }
      // click the load model button
      await page.locator('#btn-load').click();
      // wait for the model to load or fail
      await page.waitForFunction("['2','3'].includes(document.getElementById('ortstate').innerText)");
      // verify the model is loaded
      const modelLoadState = await page.$eval('#ortstate', (el) => el.innerText);
      if (modelLoadState !== '2') {
        const ortLog = await page.$eval('#ortlog', (el) => el.innerText);
        testResults.push({ multiThread, proxy, success: false, message: `Failed to load model: ${ortLog}` });
        continue;
      }

      // click the run test button
      await page.locator('#btn-run').click();
      // wait for the inference run to complete or fail
      await page.waitForFunction("['5','6'].includes(document.getElementById('ortstate').innerText)");
      // verify the inference run result
      const runState = await page.$eval('#ortstate', (el) => el.innerText);
      if (runState !== '5') {
        const ortLog = await page.$eval('#ortlog', (el) => el.innerText);
        testResults.push({ multiThread, proxy, success: false, message: `Failed to run model: ${ortLog}` });
        continue;
      }

      testResults.push({ multiThread, proxy, success: true });
    }

    return testResults;
  } finally {
    console.log(`[${logPrefix}] Closing the browser...`);
    // close the browser
    if (browser) {
      await browser.close();
    }
  }
};

async function runTest(testCaseName, logPrefix, ready, cmd, port) {
  const wd = path.join(__dirname, 'testcases', testCaseName);
  logPrefix = [testCaseName, ...logPrefix].join(':');

  console.log(`[${logPrefix}] Testing...`);
  const npmRunDevEvent = new EventEmitter();
  const npmRunDev = runShellCmd(cmd, {
    wd,
    event: npmRunDevEvent,
    ready,
    ignoreExitCode: true,
  });

  let testResults;
  npmRunDevEvent.on('serverReady', async () => {
    try {
      testResults = await launchBrowserAndRunTests(logPrefix, port);
    } finally {
      console.log(`[${logPrefix}] Killing the server...`);
      // kill the server as the last step
      npmRunDevEvent.emit('kill');
    }
  });

  await npmRunDev;

  console.log(`[${logPrefix}] test result:`, testResults);
  if (testResults.some((r) => !r.success)) {
    throw new Error(`[${logPrefix}] test failed.`);
  }
}

async function runDevTest(testCaseName, ready, port, logPrefix = '', cmd = 'npm run dev') {
  return runTest(testCaseName, logPrefix ? ['dev', logPrefix] : ['dev'], ready, cmd, port);
}

async function runProdTest(testCaseName, ready, port) {
  const wd = path.join(__dirname, 'testcases', testCaseName);

  console.log(`[${testCaseName}:prod] Building...`);
  await runShellCmd('npm run build', { wd });
  await runTest(testCaseName, ['prod'], ready, 'npm run start', port);
}

async function verifyAssets(testCaseName, testers) {
  testers = Array.isArray(testers) ? testers : [testers];
  const wd = path.join(__dirname, 'testcases', testCaseName);

  console.log(`[${testCaseName}] Verifying assets...`);

  const testResults = [];

  try {
    for (const tester of testers) {
      testResults.push(await tester(wd));
    }

    if (testResults.some((r) => !r.success)) {
      throw new Error(`[${testCaseName}] asset verification failed.`);
    }
  } finally {
    console.log(`[${testCaseName}] asset verification result:`, testResults);
  }
}

module.exports = {
  runDevTest,
  runProdTest,
  verifyAssets,
};
