// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// this is a simple HTTP server that enables CORS.
// following code is based on https://developer.mozilla.org/en-US/docs/Learn/Server-side/Node_server_without_framework

const http = require('http');
const fs = require('fs');
const path = require('path');

const getRequestData = (url, dir) => {
  const pathname = new URL(url, 'http://localhost').pathname;

  let filepath;
  let mimeType;
  if (
    pathname.startsWith('/test-wasm-path-override/') ||
    pathname.startsWith('/dist/') ||
    pathname.startsWith('/esm-loaders/')
  ) {
    filepath = path.resolve(dir, pathname.substring(1));
  } else {
    return null;
  }

  if (filepath.endsWith('.wasm')) {
    mimeType = 'application/wasm';
  } else if (filepath.endsWith('.js') || filepath.endsWith('.mjs')) {
    mimeType = 'text/javascript';
  } else {
    return null;
  }

  return [filepath, mimeType];
};

module.exports = function (dir, port) {
  const server = http
    .createServer(function (request, response) {
      const url = request.url.replace(/\n|\r/g, '');
      console.log(`request ${url}`);

      const requestData = getRequestData(url, dir);
      if (!request || !requestData) {
        response.writeHead(404);
        response.end('404');
      } else {
        const [filePath, contentType] = requestData;
        fs.readFile(path.resolve(dir, filePath), function (error, content) {
          if (error) {
            if (error.code == 'ENOENT') {
              response.writeHead(404);
              response.end('404');
            } else {
              response.writeHead(500);
              response.end('500');
            }
          } else {
            response.setHeader('access-control-allow-origin', '*');
            response.writeHead(200, { 'Content-Type': contentType });
            response.end(content, 'utf-8');
          }
        });
      }
    })
    .listen(port);
  console.log(`Server running at http://localhost:${port}/`);
  return server;
};
