// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This script generates a version number for @dev NPM packages and output it to stdout.

const fs = require('fs');
const path = require('path');
const child_process = require('child_process');

// Get current version
const VERSION_BASE = fs.readFileSync(path.join(__dirname, '../../../../VERSION_NUMBER')).toString().trim();

const now = new Date().toISOString();
const YYYY = now.slice(0, 4);
const MM = now.slice(5, 7);
const DD = now.slice(8, 10);
const hh = now.slice(11, 13);
const mm = now.slice(14, 16);

const GIT_COMMIT_ID = child_process.execSync('git rev-parse --short HEAD').toString().trim();

const DATE_STRING = `${YYYY}${MM}${DD}-${hh}${mm}`;
const VERSION = `${VERSION_BASE}-dev.${DATE_STRING}.${GIT_COMMIT_ID}`;
console.log(`##vso[task.setvariable variable=NpmPackageVersionNumber;]${VERSION}`);
