// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This script is used to download WebAssembly build artifacts from CI pipeline.
//
// The goal of this script is to save time for ORT Web developers. For most TypeScript tasks, there is no change in the
// WebAssembly side, so there is no need to rebuild WebAssembly.
//
// It performs the following operations:
// 1. query build ID for latest successful build on master branch
// 2. query download URL of build artifacts
// 3. download and unzip the files to folders
//

import fs from 'fs';
import https from 'https';
import jszip from 'jszip';
import path from 'path';

function downloadJson(url: string, onSuccess: (data: any) => void) {
  https.get(url, res => {
    const {statusCode} = res;
    const contentType = res.headers['content-type'];

    if (statusCode !== 200) {
      throw new Error(`Failed to download build list. HTTP status code = ${statusCode}`);
    }
    if (!contentType || !/^application\/json/.test(contentType)) {
      throw new Error(`unexpected content type: ${contentType}`);
    }
    res.setEncoding('utf8');
    let rawData = '';
    res.on('data', (chunk) => {
      rawData += chunk;
    });
    res.on('end', () => {
      onSuccess(JSON.parse(rawData));
    });
  });
}

function downloadZip(url: string, onSuccess: (data: Buffer) => void) {
  https.get(url, res => {
    const {statusCode} = res;
    const contentType = res.headers['content-type'];

    if (statusCode !== 200) {
      throw new Error(`Failed to download build list. HTTP status code = ${statusCode}`);
    }
    if (!contentType || !/^application\/zip/.test(contentType)) {
      throw new Error(`unexpected content type: ${contentType}`);
    }

    const chunks: Buffer[] = [];
    res.on('data', (chunk) => {
      chunks.push(chunk);
    });
    res.on('end', () => {
      onSuccess(Buffer.concat(chunks));
    });
  });
}

function extractFile(zip: jszip, folder: string, file: string, artifactName: string) {
  zip.file(`${artifactName}/${file}`)!.nodeStream()
      .pipe(fs.createWriteStream(path.join(folder, file)))
      .on('finish', () => {
        console.log('# file downloaded and extracted: ' + file);
      });
}

console.log('=== Start to pull WebAssembly artifacts from CI ===');

// API reference: https://docs.microsoft.com/en-us/rest/api/azure/devops/build/builds/list
downloadJson(
    'https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/builds?api-version=6.1-preview.6' +
        '&definitions=161' +
        '&resultFilter=succeeded' +
        '&$top=1' +
        '&repositoryId=Microsoft/onnxruntime' +
        '&repositoryType=GitHub' +
        '&branchName=refs/heads/master',
    data => {
      const buildId = data.value[0].id;

      console.log(`=== Found latest master build : ${buildId} ===`);

      // API reference: https://docs.microsoft.com/en-us/rest/api/azure/devops/build/artifacts/get%20artifact
      downloadJson(
          `https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/builds/${
              buildId}/artifacts?api-version=6.1-preview.5`,
          data => {
            let ortWasmZipLink, ortWasmThreadedZipLink;
            for (const v of data.value) {
              if (v.name === 'Release_ort-wasm') {
                ortWasmZipLink = v.resource.downloadUrl;
              }
              if (v.name === 'Release_ort-wasm-threaded') {
                ortWasmThreadedZipLink = v.resource.downloadUrl;
              }
            }

            console.log('=== Ready to download zip files ===');

            const WASM_FOLDER = path.join(__dirname, '../dist');
            if (!fs.existsSync(WASM_FOLDER)) {
              fs.mkdirSync(WASM_FOLDER);
            }
            const JS_FOLDER = path.join(__dirname, '../lib/wasm/binding');

            downloadZip(ortWasmZipLink, buffer => {
              void jszip.loadAsync(buffer).then(zip => {
                extractFile(zip, JS_FOLDER, 'ort-wasm.js', 'Release_ort-wasm');
                extractFile(zip, WASM_FOLDER, 'ort-wasm.wasm', 'Release_ort-wasm');
              });
            });

            downloadZip(ortWasmThreadedZipLink, buffer => {
              void jszip.loadAsync(buffer).then(zip => {
                extractFile(zip, JS_FOLDER, 'ort-wasm-threaded.js', 'Release_ort-wasm-threaded');
                extractFile(zip, JS_FOLDER, 'ort-wasm-threaded.worker.js', 'Release_ort-wasm-threaded');
                extractFile(zip, WASM_FOLDER, 'ort-wasm-threaded.wasm', 'Release_ort-wasm-threaded');
              });
            });
          });
    });
