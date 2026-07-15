// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const { EOL } = require("os");
const readline = require("readline");

// This script is used to parse the Chromium debug log and extract the raw log data.

async function processChromiumDebugLog() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const result =
      /^\[.+INFO:CONSOLE.+?]\ "(?<raw_log_data>.+)",\ source:\ [^"]+?\(\d+\)$/.exec(
        line
      );
    if (!result) {
      continue;
    }
    const rawLogData = result.groups.raw_log_data;
    process.stdout.write(`${rawLogData}${EOL}`);
  }
}

processChromiumDebugLog();
