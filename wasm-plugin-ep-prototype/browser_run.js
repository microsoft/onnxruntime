// Headless-browser harness for the wasm plugin-EP prototype.
//
// Usage: node browser_run.js <build-dir> <mode>[,<mode>...]
//   modes: preload | dlopen | ondemand
//
// Serves the build dir with COOP/COEP (so -pthread / SharedArrayBuffer works) and drives
// Chromium through Playwright, printing everything the page logged.

const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const buildDir = path.resolve(process.argv[2] || '.');
const modes = (process.argv[3] || 'preload').split(',');
const harness = path.join(__dirname, 'browser_test.html');

const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  const file = (url === '/' || url === '/index.html')
    ? harness
    : path.join(buildDir, path.normalize(url).replace(/^[\\/]+/, ''));
  fs.readFile(file, (err, data) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    if (err) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

(async () => {
  await new Promise((r) => server.listen(0, r));
  const port = server.address().port;

  const browser = await chromium.launch({
    args: [
      '--no-sandbox',
      '--enable-features=SharedArrayBuffer',
      '--enable-experimental-webassembly-features',
      '--js-flags=--experimental-wasm-jspi',
      '--enable-unsafe-swiftshader',
    ],
  });

  const version = browser.version();
  console.log(`chromium ${version}`);

  let failures = 0;
  for (const mode of modes) {
    console.log(`\n########## mode=${mode} (${path.basename(buildDir)}) ##########`);
    const page = await browser.newPage();
    page.on('console', (m) => console.log(`  ${m.text()}`));
    page.on('pageerror', (e) => console.log(`  PAGEERROR ${e.message}`));

    let status = 'TIMEOUT';
    try {
      await page.goto(`http://localhost:${port}/?mode=${mode}&synclimit=1`, { waitUntil: 'load' });
      status = await page.waitForFunction(
        () => window.__result || null, null, { timeout: 60000 },
      ).then((h) => h.jsonValue());
    } catch (e) {
      console.log(`  harness error: ${e.message}`);
    }
    const log = await page.evaluate(() => document.getElementById('log').textContent);
    console.log(log.split('\n').map((l) => '  ' + l).join('\n'));
    console.log(`  ==> STATUS=${status}`);
    if (status !== 'PASS') failures++;
    await page.close();
  }

  await browser.close();
  server.close();
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
