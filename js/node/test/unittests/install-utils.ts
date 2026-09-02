// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as assert from 'assert';
import * as fs from 'fs';
import * as http from 'http';
import * as os from 'os';
import * as path from 'path';

const { downloadFile, downloadJson } = require('../../script/install-utils.js') as {
  downloadFile: (url: string, destination: string) => Promise<void>;
  downloadJson: (url: string) => Promise<unknown>;
};

describe('#UnitTest# - install utility downloads', () => {
  let server: http.Server;
  let baseUrl: string;

  before((done) => {
    server = http.createServer((request, response) => {
      if (request.url === '/redirect-json') {
        response.writeHead(302, { Location: '/metadata.json' }).end();
        return;
      }
      if (request.url === '/redirect-file') {
        response.writeHead(307, { Location: '/package.bin' }).end();
        return;
      }
      if (request.url === '/redirect-loop') {
        response.writeHead(302, { Location: '/redirect-loop' }).end();
        return;
      }
      if (request.url === '/redirect-malformed') {
        response.writeHead(302, { Location: 'http://[invalid' }).end();
        return;
      }
      if (request.url === '/metadata.json') {
        response.writeHead(200, { 'Content-Type': 'application/json' }).end('{"version":"test"}');
        return;
      }
      if (request.url === '/package.bin') {
        response.writeHead(200, { 'Content-Type': 'application/octet-stream' }).end('package contents');
        return;
      }
      response.writeHead(404).end();
    });
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      assert.ok(address && typeof address !== 'string');
      baseUrl = `http://127.0.0.1:${address.port}`;
      done();
    });
  });

  after((done) => server.close(done));

  it('follows redirects when downloading JSON metadata', async () => {
    assert.deepStrictEqual(await downloadJson(`${baseUrl}/redirect-json`), { version: 'test' });
  });

  it('follows redirects when downloading package files', async () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnxruntime-install-utils-'));
    const destination = path.join(tempDir, 'package.bin');
    try {
      await downloadFile(`${baseUrl}/redirect-file`, destination);
      assert.strictEqual(fs.readFileSync(destination, 'utf8'), 'package contents');
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('rejects when the redirect limit is exceeded', async () => {
    await assert.rejects(downloadJson(`${baseUrl}/redirect-loop`), /Too many redirects/);
  });

  it('rejects malformed redirect locations', async () => {
    await assert.rejects(downloadJson(`${baseUrl}/redirect-malformed`), /Invalid URL/);
  });
});
