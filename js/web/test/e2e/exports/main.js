// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const { spawn } = require('child_process');
const EventEmitter = require('node:events');
const treeKill = require('tree-kill');

/**
 * Entry point for package exports tests.
 *
 * @param {string[]} packagesToInstall
 */
module.exports = async function main(PRESERVE, PACKAGES_TO_INSTALL) {
  console.log('Running exports tests...');

  // testcases/nextjs-default
  {
    const wd = path.join(__dirname, 'testcases/nextjs-default');
    if (!PRESERVE) {
      if (PACKAGES_TO_INSTALL.length === 0) {
        await runShellCmd('npm ci', { wd });
      } else {
        await runShellCmd(`npm install ${PACKAGES_TO_INSTALL.map((i) => `"${i}"`).join(' ')}`, { wd });
      }
    }

    const launchBrowserAndRunTests = async (logPrefix, port = 3000) => {
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

    // test dev mode
    {
      console.log('Testing Next.js default (dev mode)...');
      const npmRunDevEvent = new EventEmitter();
      const npmRunDev = runShellCmd('npm run dev', {
        wd,
        event: npmRunDevEvent,
        ready: '✓ Ready in',
        ignoreExitCode: true,
      });

      let testResults;
      npmRunDevEvent.on('serverReady', async () => {
        try {
          testResults = await launchBrowserAndRunTests('default:dev');
        } finally {
          console.log('Killing the server...');
          // kill the server as the last step
          npmRunDevEvent.emit('kill');
        }
      });

      await npmRunDev;

      console.log('Next.js default test (dev mode) result:', testResults);
      if (testResults.some((r) => !r.success)) {
        throw new Error('Next.js default test (dev mode) failed.');
      }
    } // test dev mode

    // test dev mode (Turbopack)
    {
      console.log('Testing Next.js default (dev mode with turbopack)...');
      const npmRunDevEvent = new EventEmitter();
      const npmRunDev = runShellCmd('npm run dev -- --turbopack', {
        wd,
        event: npmRunDevEvent,
        ready: '✓ Ready in',
        ignoreExitCode: true,
      });

      let testResults;
      npmRunDevEvent.on('serverReady', async () => {
        try {
          testResults = await launchBrowserAndRunTests('default:dev:turbopack');
        } finally {
          console.log('Killing the server...');
          // kill the server as the last step
          npmRunDevEvent.emit('kill');
        }
      });

      await npmRunDev;

      console.log('Next.js default test (dev mode with turbopack) result:', testResults);
      if (testResults.some((r) => !r.success)) {
        throw new Error('Next.js default test (dev mode with turbopack) failed.');
      }
    } // test dev mode

    // test prod mode
    {
      console.log('Testing Next.js default (prod mode)...');
      // run 'npm run build'
      await runShellCmd('npm run build', { wd });
      const npmRunStartEvent = new EventEmitter();
      const npmRunStart = runShellCmd('npm run start', {
        wd,
        event: npmRunStartEvent,
        ready: '✓ Ready in',
        ignoreExitCode: true,
      });

      let testResults;
      npmRunStartEvent.on('serverReady', async () => {
        try {
          testResults = await launchBrowserAndRunTests('default:prod');
        } finally {
          console.log('Killing the server...');
          // kill the server as the last step
          npmRunStartEvent.emit('kill');
        }
      });

      await npmRunStart;

      console.log('Next.js default test (prod mode) result:', testResults);
      if (testResults.some((r) => !r.success)) {
        throw new Error('Next.js default test (prod mode) failed.');
      }
    } // test prod mode
  }
};

async function runShellCmd(cmd, { wd = __dirname, event = null, ready = null, ignoreExitCode = false }) {
  console.log('===============================================================');
  console.log(' Running command in shell:');
  console.log(' > ' + cmd);
  console.log('===============================================================');

  return new Promise((resolve, reject) => {
    const childProcess = spawn(cmd, { shell: true, stdio: ['ignore', 'pipe', 'inherit'], cwd: wd });
    childProcess.on('close', function (code, signal) {
      if (code === 0 || ignoreExitCode) {
        resolve();
      } else {
        reject(`Process exits with code ${code}`);
      }
    });
    childProcess.stdout.on('data', (data) => {
      process.stdout.write(data);

      if (ready && event && data.toString().includes(ready)) {
        event.emit('serverReady');
      }
    });
    if (event) {
      event.on('kill', () => {
        childProcess.stdout.destroy();
        treeKill(childProcess.pid);
        console.log('killing process...');
      });
    }
  });
}
