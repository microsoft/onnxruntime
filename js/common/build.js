// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import {execSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// build the following folders:
// - dist/cjs
// - dist/esm
execSync('npm run build:cjs', {shell: true, stdio: 'inherit', cwd: __dirname});
execSync('npm run build:esm', {shell: true, stdio: 'inherit', cwd: __dirname});

// generate package.json files under each of the dist folders for commonJS and ESModule
// this trick allows typescript to import this package as different module type
// see also: https://evertpot.com/universal-commonjs-esm-typescript-packages/
writeFileSync(resolve(__dirname, './dist/cjs', 'package.json'), '{"type": "commonjs"}');
writeFileSync(resolve(__dirname, './dist/esm', 'package.json'), '{"type": "module"}');
