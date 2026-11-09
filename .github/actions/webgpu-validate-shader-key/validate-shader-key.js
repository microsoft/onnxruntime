// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const { EOL } = require("os");
const readline = require("readline");

// This script is used to parse the raw log data and check if there are inconsistent shader.
// Within the same device configuration, the same shader key must produce the same code.

const shaderMap = new Map();

const regexShaderStart =
  /^===\ WebGPU\ Shader\ code\ \[.+?, Configuration="(?<configuration>[^"]+)", Key="(?<key>.+)"]\ Start\ ===$/;
const regexShaderEnd =
  /^===\ WebGPU\ Shader\ code\ \[.+?, Configuration="(?<configuration>[^"]+)", Key="(?<key>.+)"]\ End\ ===$/;

async function processVerboseLog() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  let currentShaderKey = null;
  let currentShaderConfiguration = null;
  let currentShaderCode = null;
  let currentShaderStartLineNumber = null;
  let lineNumber = 0;

  for await (const line of rl) {
    lineNumber++;
    const resultStart = regexShaderStart.exec(line);
    if (resultStart) {
      if (currentShaderKey) {
        throw new Error(
          `Line ${lineNumber}: Found incomplete shader code for key "${currentShaderKey}" (shader started at line ${currentShaderStartLineNumber}).`
        );
      }

      currentShaderKey = resultStart.groups.key;
      currentShaderConfiguration = resultStart.groups.configuration;
      currentShaderCode = "";
      currentShaderStartLineNumber = lineNumber;
      continue;
    }

    const resultEnd = regexShaderEnd.exec(line);
    if (resultEnd) {
      if (!currentShaderKey) {
        throw new Error(
          `Line ${lineNumber}: Found unexpected shader end for key "${resultEnd.groups.key}".`
        );
      }

      if (
        resultEnd.groups.key !== currentShaderKey ||
        resultEnd.groups.configuration !== currentShaderConfiguration
      ) {
        throw new Error(
          `Line ${lineNumber}: Shader end key or configuration does not match its start at line ${currentShaderStartLineNumber}.`
        );
      }

      const comparisonKey = JSON.stringify([
        currentShaderConfiguration,
        currentShaderKey,
      ]);
      if (shaderMap.has(comparisonKey)) {
        if (shaderMap.get(comparisonKey) !== currentShaderCode) {
          throw new Error(`Line ${lineNumber}: Found inconsistent shader code for key "${currentShaderKey}" (configuration ${JSON.stringify(currentShaderConfiguration)}, shader started at line ${currentShaderStartLineNumber}).
=== Previous Shader Start ===
${shaderMap.get(comparisonKey)}
=== Previous Shader End ===

=== Current Shader Start ===
${currentShaderCode}
=== Current Shader End ===`);
        }
      } else {
        shaderMap.set(comparisonKey, currentShaderCode);
      }

      currentShaderKey = null;
      currentShaderConfiguration = null;
      currentShaderCode = null;
      currentShaderStartLineNumber = null;
      continue;
    }

    if (line.startsWith("=== WebGPU Shader code [")) {
      throw new Error(
        `Line ${lineNumber}: Shader header must include nonempty Configuration and Key fields.`
      );
    }

    if (currentShaderKey) {
      currentShaderCode += line + EOL;
    }
  }

  if (currentShaderKey) {
    throw new Error(
      `Reached end of log (line ${lineNumber}) with incomplete shader code for key "${currentShaderKey}" (shader started at line ${currentShaderStartLineNumber}).`
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
