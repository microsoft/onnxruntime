// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


/* eslint-disable @typescript-eslint/restrict-plus-operands */

// parse-profiler
//
// this script is used to parse performance profiling result.
// usage:
// STEP.1 - profiling
// > npm test -- model test/test-data/{path-to-my-model} --backend={cpu/webgl/wasm} --profile > profile.raw.log
// STEP.2 - parse
// > node tools/parse-profiler < profile.raw.log > profile.parsed.log


import * as readline from 'readline';
const int = readline.createInterface({input: process.stdin, output: process.stdout, terminal: false});

// eslint-disable-next-line no-control-regex
const matcher = /Profiler\.([^[\s\x1b]+)(\x1b\[0m)? (\d.+Z)\|([\d.]+)ms on event '([^']+)' at (\d*\.*\d*)/;

const allEvents: any[] = [];
int.on('line', input => {
  const matches = matcher.exec(input);
  if (matches) {
    // console.log(matches);
    const category = matches[1];
    const logTimeStamp = new Date(matches[3]);
    const ms = Number.parseFloat(matches[4]);
    const event = matches[5];
    const endTimeInNumber = matches[6];
    allEvents.push({event, ms, logTimeStamp, category, endTimeInNumber});
  }
});

int.on('close', () => {
  for (const i of allEvents) {
    console.log(`${(i.category + '           ').substring(0, 12)} ${((i.ms) + '           ').substring(0, 12)} ${
        (i.event + '                                      ').substring(0, 40)} ${i.endTimeInNumber}`);
  }
});
