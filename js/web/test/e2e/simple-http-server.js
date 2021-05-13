// following code is based on https://developer.mozilla.org/en-US/docs/Learn/Server-side/Node_server_without_framework

var http = require('http');
var fs = require('fs');
var path = require('path');

module.exports = function (dir) {
  http.createServer(function (request, response) {
    console.log('request ', request.url);

    var filePath = '.' + request.url;

    var extname = String(path.extname(filePath)).toLowerCase();
    var mimeTypes = {
      '.html': 'text/html',
      '.js': 'text/javascript',
      '.css': 'text/css',
      '.json': 'application/json',
      '.png': 'image/png',
      '.jpg': 'image/jpg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml',
      '.wav': 'audio/wav',
      '.mp4': 'video/mp4',
      '.woff': 'application/font-woff',
      '.ttf': 'application/font-ttf',
      '.eot': 'application/vnd.ms-fontobject',
      '.otf': 'application/font-otf',
      '.wasm': 'application/wasm'
    };

    var contentType = mimeTypes[extname] || 'application/octet-stream';

    fs.readFile(path.resolve(dir, filePath), function (error, content) {
      if (error) {
        if (error.code == 'ENOENT') {
          response.writeHead(404);
          response.end('404');
        }
        else {
          response.writeHead(500);
          response.end('500');
        }
      }
      else {
        response.setHeader('access-control-allow-origin', '*');
        response.writeHead(200, { 'Content-Type': contentType });
        response.end(content, 'utf-8');
      }
    });

  }).listen(8081);
  console.log('Server running at http://127.0.0.1:8081/');
};
