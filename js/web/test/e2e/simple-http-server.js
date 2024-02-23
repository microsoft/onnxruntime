// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// this is a simple HTTP server that enables CORS.
// following code is based on https://developer.mozilla.org/en-US/docs/Learn/Server-side/Node_server_without_framework

const http = require('http');
const fs = require('fs');
const path = require('path');

const validRequests = {
  // .wasm files
  '/dist/ort-wasm.wasm': ['dist/ort-wasm.wasm', 'application/wasm'],
  '/dist/ort-wasm-simd.wasm': ['dist/ort-wasm-simd.wasm', 'application/wasm'],
  '/dist/ort-wasm-threaded.wasm': ['dist/ort-wasm-threaded.wasm', 'application/wasm'],
  '/dist/ort-wasm-simd-threaded.wasm': ['dist/ort-wasm-simd-threaded.wasm', 'application/wasm'],
  '/dist/ort-wasm-simd.jsep.wasm': ['dist/ort-wasm-simd.jsep.wasm', 'application/wasm'],

  // proxied .wasm files:
  '/test-wasm-path-override/ort-wasm.wasm': ['dist/ort-wasm.wasm', 'application/wasm'],
  //'/test-wasm-path-override/renamed.wasm': ['dist/ort-wasm.wasm', 'application/wasm'],

  // .js files
  '/dist/ort.min.js': ['dist/ort.min.js', 'text/javascript'],
  '/dist/ort.js': ['dist/ort.js', 'text/javascript'],
  '/dist/ort.webgl.min.js': ['dist/ort.webgl.min.js', 'text/javascript'],
  '/dist/ort.webgpu.min.js': ['dist/ort.webgpu.min.js', 'text/javascript'],
  '/dist/ort.wasm.min.js': ['dist/ort.wasm.min.js', 'text/javascript'],
  '/dist/ort.wasm-core.min.js': ['dist/ort.wasm-core.min.js', 'text/javascript'],
};

module.exports = function(dir) {
  http.createServer(function(request, response) {
        console.log(`request ${request.url.replace(/\n|\r/g, '')}`);

        const requestData = validRequests[request.url];
        if (!request) {
          response.writeHead(404);
          response.end('404');
        } else {
          const [filePath, contentType] = requestData;
          fs.readFile(path.resolve(dir, filePath), function(error, content) {
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
              response.writeHead(200, {'Content-Type': contentType});
              response.end(content, 'utf-8');
            }
          });
        }
      })
      .listen(8081);
  console.log('Server running at http://127.0.0.1:8081/');
};
