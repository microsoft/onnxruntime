// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const { EOL } = require("os");
const readline = require("readline");

// This script is used to parse the raw log data and check if there are inconsistent shader.
// When the shader key is the same, the shader code should be the same.

const shaderMap = new Map();

const regexStartingProgram =
  /onnxruntime::webgpu::WebGpuContext::Run.+Starting program \"(?<key>.+)\"/;
const regexShaderStart =
  /^===\ WebGPU\ Shader\ code\ \[.+?(Key=\"(?<key>.+)\")?]\ Start\ ===$/;
const regexShaderEnd =
  /^===\ WebGPU\ Shader\ code\ \[.+?(Key=\"(?<key>.+)\")?]\ End\ ===$/;

async function processVerboseLog() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  let lastProgramKey = null;
  let currentShaderKey = null;
  let currentShaderCode = null;

  for await (const line of rl) {
    const startingProgram = regexStartingProgram.exec(line);
    if (startingProgram) {
      lastProgramKey = startingProgram.groups.key;
      continue;
    }

    const resultStart = regexShaderStart.exec(line);
    if (resultStart) {
      if (currentShaderKey) {
        throw new Error(
          `Found incomplete shader code for key "${currentShaderKey}".`
        );
      }

      const key = resultStart.groups.key ?? lastProgramKey;
      if (!key) {
        throw new Error(
          'No shader key is found in the log. Please use debug build or enable verbose logging in session options in release build.'
        );
      }
      if (lastProgramKey && key !== lastProgramKey) {
        throw new Error(
          `Found incorrect shader key from log. Expected "${lastProgramKey}", but got "${key}".`
        );
      }
      currentShaderKey = key;
      currentShaderCode = "";
      continue;
    }

    const resultEnd = regexShaderEnd.exec(line);
    if (resultEnd) {
      if (!currentShaderKey) {
        throw new Error(
          `Found unexpected shader end for key "${resultEnd.groups.key}".`
        );
      }

      const key = resultEnd.groups.key ?? lastProgramKey;
      if (!key) {
        throw new Error(
          'No shader key is found in the log. Please use debug build or enable verbose logging in session options in release build.'
        );
      }
      if (lastProgramKey && key !== lastProgramKey) {
        throw new Error(
          `Found incorrect shader key from log. Expected "${lastProgramKey}", but got "${key}".`
        );
      }

      if (shaderMap.has(currentShaderKey)) {
        if (shaderMap.get(currentShaderKey) !== currentShaderCode) {
          throw new Error(`Found inconsistent shader code for key "${currentShaderKey}".
=== Previous Shader Start ===
${shaderMap.get(currentShaderKey)}
=== Previous Shader End ===

=== Current Shader Start ===
${currentShaderCode}
=== Current Shader End ===`);
        }
      } else {
        shaderMap.set(currentShaderKey, currentShaderCode);
      }

      currentShaderKey = null;
      currentShaderCode = null;
      continue;
    }

    if (currentShaderKey) {
      currentShaderCode += line + EOL;
    }
  }

  if (currentShaderKey) {
    throw new Error(
      `Found incomplete shader code for key "${currentShaderKey}".`
    );
  }

  if (shaderMap.size === 0) {
    throw new Error("No shader code found.");
  }

  console.log(
    `All shader code is consistent. Total ${shaderMap.size} shader keys found.`
  );
}

processVerboseLog();
