// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WriteStream } from 'fs';
import { bootstrap as globalAgentBootstrap } from 'global-agent';
import * as https from 'https';
import { JSZipObject } from 'jszip';

// Bootstrap global-agent to honor the proxy settings in
// environment variables, e.g. GLOBAL_AGENT_HTTPS_PROXY.
// See https://github.com/gajus/global-agent/blob/v3.0.0/README.md#environment-variables for details.
globalAgentBootstrap();

export const downloadZip = async (url: string, maxRetryTimes = 3): Promise<Buffer> => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetryTimes; attempt++) {
    try {
      return await new Promise<Buffer>((resolve, reject) => {
        https
          .get(url, (res) => {
            const { statusCode } = res;
            const contentType = res.headers['content-type'];

            if (statusCode === 301 || statusCode === 302) {
              downloadZip(res.headers.location!, maxRetryTimes).then(
                (buffer) => resolve(buffer),
                (reason) => reject(reason),
              );
              return;
            } else if (statusCode !== 200) {
              reject(new Error(`Failed to download build list. HTTP status code = ${statusCode}`));
              return;
            }
            if (!contentType || !/^application\/zip/.test(contentType)) {
              reject(new Error(`unexpected content type: ${contentType}`));
              return;
            }

            const chunks: Buffer[] = [];
            res.on('data', (chunk) => {
              chunks.push(chunk);
            });
            res.on('end', () => {
              resolve(Buffer.concat(chunks));
            });
            res.on('error', (err) => {
              reject(err);
            });
          })
          .on('error', (err) => {
            reject(err);
          });
      });
    } catch (error) {
      lastError = error;
      if (attempt < maxRetryTimes) {
        // Wait before retrying (exponential backoff)
        const delay = Math.pow(2, attempt) * 10000; // 10s, 20s, 40s, etc.
        await new Promise((resolve) => setTimeout(resolve, delay));
        console.warn(`Download attempt ${attempt + 1} failed, retrying in ${delay}ms...`);
      }
    }
  }

  throw lastError || new Error(`Failed to download after ${maxRetryTimes + 1} attempts`);
};

export const extractFile = async (entry: JSZipObject, ostream: WriteStream): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    entry
      .nodeStream()
      .pipe(ostream)
      .on('finish', () => {
        resolve();
      })
      .on('error', (err) => {
        reject(err);
      });
  });
