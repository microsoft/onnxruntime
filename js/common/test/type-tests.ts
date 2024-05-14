// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import globby from 'globby';
import assert from 'node:assert';
import {readFileSync} from 'node:fs';
import {dirname, join, normalize, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import npmlog from 'npmlog';
import typescript from 'typescript';

/**
 * @fileoverview
 *
 * This file is used to run TypeScript type tests.
 *
 * The type tests are located under `common/test/type-tests/` folder. Each test file is a simple typescrit source file.
 * In each test file, any expected failure are marked as comments in the following format:
 *
 * "// {type-tests}|fail|<line-number-offset>|<error-code>"
 *
 * - comments should always start with "// {type-tests}|fail|"
 * - <line-number-offset> is the line number offset from the comment line
 * - <error-code> is the error code of the expected failure
 *
 * For example:
 *
 * ```ts
 * // {type-tests}|fail|1|2348
 * const t0 = ort.Tensor('xxx');
 * ```
 *
 * In this example, the comments indicate that the line number of the expected failure is 1 line after the comment line,
 * and the error code is 2348.
 *
 * The test runner will compile each test file and check if the actual error code matches the expected error code. If
 * there is no expected failure, the test runner will check if the test file can be compiled successfully.
 */

// the root folder of type tests
const TYPE_TESTS_DIR = join(dirname(fileURLToPath(import.meta.url)), './type-tests');

/**
 * aggregate test files: `*.ts` under folder `common/test/type-tests/`
 *
 * @returns list of test files
 */
const prepareTestFileList = () =>
    //
    globby.sync('**/*.ts', {
      cwd: TYPE_TESTS_DIR,
      absolute: true,
    });

/**
 * Run typescript compiler on the given files.
 */
const compileTypeScriptFiles = (filepaths: string[]): readonly typescript.Diagnostic[] => {
  // TypeScript compiler options, base URL is reset to `TYPE_TESTS_DIR`.
  const compilerOptions =
      JSON.parse(readFileSync(new URL('./type-tests/tsconfig.json', import.meta.url), 'utf-8')).compilerOptions as
      typescript.CompilerOptions;
  compilerOptions.baseUrl = TYPE_TESTS_DIR;

  // Run TypeScript compiler
  const program = typescript.createProgram({
    rootNames: filepaths,
    options: compilerOptions,
  });

  return typescript.getPreEmitDiagnostics(program);
};

/**
 * Prepare test cases for TypeScript type tests.
 * @returns list of test cases. Each test case data contains the test title and a function to run the test.
 */
const prepareTestCases = () => {
  npmlog.info('PrepareTestCases', 'Preparing test file lists...');
  const testFiles = prepareTestFileList();
  npmlog.info('PrepareTestCases', `Preparing test file lists... DONE, ${testFiles.length} file(s) in total.`);

  npmlog.info('PrepareTestCases', 'Running TypeScript Compiler...');
  const compileResult = compileTypeScriptFiles(testFiles).map(
      diagnostic => ({
        fileName: normalize(diagnostic.file?.fileName ?? ''),
        line: diagnostic.file?.getLineAndCharacterOfPosition(diagnostic.start!)?.line ?? -1,
        code: diagnostic.code,
      }));
  npmlog.info('PrepareTestCases', 'Running TypeScript Compiler... DONE.');

  npmlog.info('PrepareTestCases', 'Parsing test source files for expected failures...');
  const testCases = testFiles.map(filepath => {
    const normalizedFilePath = normalize(filepath);
    const normalizedRelativePath = normalize(relative(TYPE_TESTS_DIR, filepath));

    const fileAllLines = readFileSync(filepath, 'utf-8').split('\n').map(line => line.trim());
    const expectedFailures: Array<{line: number; code: number}> = [];
    fileAllLines.forEach((line, i) => {
      if (line.startsWith('// {type-tests}|fail|')) {
        const splitted = line.split('|');
        assert(splitted.length === 4, `invalid expected failure comment: ${line}`);
        const lineOffset = Number.parseInt(splitted[2], 10);
        const code = Number.parseInt(splitted[3], 10);
        expectedFailures.push({line: i + lineOffset, code});
      }
    });

    const actualFailures: typeof compileResult = [];

    return {filepath: normalizedFilePath, relativePath: normalizedRelativePath, expectedFailures, actualFailures};
  });
  npmlog.info('PrepareTestCases', 'Parsing test source files for expected failures... DONE.');

  // now check if file names is matched
  const filePathToTestCaseMap = new Map(testCases.map(testCase => [testCase.filepath, testCase]));
  for (const error of compileResult) {
    // check file name exists
    assert(error.fileName, 'Each compile error should have a file name. Please check TypeScript compiler options.');

    // check file name is in test cases
    const testCase = filePathToTestCaseMap.get(error.fileName);
    assert(testCase, `unexpected error file name: ${error.fileName}`);

    testCase.actualFailures.push(error);
  }

  return testCases.map(testCase => {
    const {relativePath, expectedFailures, actualFailures} = testCase;
    const testFunction = () => {
      if (expectedFailures.length === 0) {
        assert.equal(actualFailures.length, 0, `expected to pass but failed: ${JSON.stringify(actualFailures)}`);
      } else {
        actualFailures.forEach(error => {
          const {line, code} = error;
          const foundIndex = expectedFailures.findIndex(f => f.line === line && f.code === code);
          assert.notEqual(foundIndex, -1, `unexpected failure: line=${line}, code=${code}`);
          expectedFailures.splice(foundIndex, 1);
        });
        assert.equal(expectedFailures.length, 0, `expected to fail but passed: ${JSON.stringify(expectedFailures)}`);
      }
    };

    return {title: relativePath, testBody: testFunction};
  });
};

describe('TypeScript type tests', () => {
  for (const {title, testBody} of prepareTestCases()) {
    it(title, testBody);
  }
});
