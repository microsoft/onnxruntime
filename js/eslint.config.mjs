// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import typescriptEslint from '@typescript-eslint/eslint-plugin';
import preferArrow from 'eslint-plugin-prefer-arrow';
import header from 'eslint-plugin-header';
import _import from 'eslint-plugin-import';
import unicorn from 'eslint-plugin-unicorn';
import jsdoc from 'eslint-plugin-jsdoc';
import { fixupPluginRules } from '@eslint/compat';
import tsParser from '@typescript-eslint/parser';
import globals from 'globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import js from '@eslint/js';
import { FlatCompat } from '@eslint/eslintrc';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

// eslint-plugin-header does not support ESLint 9 yet, but the following workaround is available:
// https://github.com/Stuk/eslint-plugin-header/issues/57#issuecomment-2378485611
header.rules.header.meta.schema = false;

export default [
  {
    ignores: [
      '**/*.js',
      '**/*.mjs',
      'eslint.config.mjs',
      '**/node_modules/',
      '**/ort-schema/',
      'common/test/type-tests/',
      'web/types.d.ts',
      'test/data/',
      '**/dist/',
    ],
  },
  ...compat.extends(
    'eslint:recommended',
    'plugin:@typescript-eslint/eslint-recommended',
    'plugin:@typescript-eslint/recommended',
  ),
  {
    plugins: {
      '@typescript-eslint': typescriptEslint,
      'prefer-arrow': preferArrow,
      header,
      import: fixupPluginRules(_import),
      unicorn,
      jsdoc,
    },

    languageOptions: {
      globals: {},
      parser: tsParser,
      ecmaVersion: 5,
      sourceType: 'module',

      parserOptions: {
        project: true,
      },
    },

    rules: {
      'unicorn/filename-case': 'error',

      'header/header': [
        2,
        'line',
        [' Copyright (c) Microsoft Corporation. All rights reserved.', ' Licensed under the MIT License.'],
        2,
      ],

      'import/no-extraneous-dependencies': [
        'error',
        {
          devDependencies: false,
        },
      ],

      'import/no-internal-modules': [
        'error',
        {
          allow: ['**/lib/**'],
        },
      ],

      'import/no-unassigned-import': 'error',

      '@typescript-eslint/array-type': [
        'error',
        {
          default: 'array-simple',
        },
      ],

      '@typescript-eslint/await-thenable': 'error',

      '@typescript-eslint/no-empty-object-type': 'error',

      '@typescript-eslint/no-wrapper-object-types': 'error',

      '@typescript-eslint/naming-convention': 'error',
      '@typescript-eslint/consistent-type-assertions': 'error',

      '@typescript-eslint/no-empty-function': 'error',
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-for-in-array': 'error',
      '@typescript-eslint/no-inferrable-types': 'error',
      '@typescript-eslint/no-misused-new': 'error',

      '@typescript-eslint/no-namespace': [
        'error',
        {
          allowDeclarations: true,
        },
      ],

      '@typescript-eslint/no-non-null-assertion': 'off',

      '@typescript-eslint/no-require-imports': [
        'error',
        {
          allow: ['^node:'],
        },
      ],

      '@typescript-eslint/no-var-requires': [
        'error',
        {
          allow: ['^node:'],
        },
      ],

      '@typescript-eslint/no-unnecessary-type-assertion': 'error',

      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
        },
      ],

      '@typescript-eslint/promise-function-async': 'error',
      '@typescript-eslint/restrict-plus-operands': 'error',

      '@typescript-eslint/triple-slash-reference': [
        'error',
        {
          path: 'always',
          types: 'prefer-import',
          lib: 'always',
        },
      ],

      'arrow-body-style': 'error',
      camelcase: 'error',
      'constructor-super': 'error',
      curly: 'error',
      'default-case': 'error',
      'dot-notation': 'error',
      eqeqeq: ['error', 'smart'],
      'guard-for-in': 'error',
      'id-match': 'error',
      'new-parens': 'error',
      'no-bitwise': 'error',
      'no-caller': 'error',
      'no-cond-assign': 'error',
      'no-console': 'error',
      'no-constant-condition': 'error',
      'no-control-regex': 'error',
      'no-debugger': 'error',
      'no-duplicate-case': 'error',
      'no-empty': 'error',
      'no-eval': 'error',
      'no-extra-bind': 'error',
      'no-invalid-regexp': 'error',
      'no-invalid-this': 'error',
      'no-multiple-empty-lines': 'error',
      'no-new-func': 'error',
      'no-new-wrappers': 'error',
      'no-octal': 'error',
      'no-octal-escape': 'error',
      'no-param-reassign': 'error',
      'no-redeclare': 'off',
      '@typescript-eslint/no-redeclare': ['error'],
      'no-regex-spaces': 'error',
      'no-return-await': 'error',
      'no-sparse-arrays': 'error',
      'no-template-curly-in-string': 'error',
      'no-throw-literal': 'error',
      'no-undef-init': 'error',
      'no-underscore-dangle': 'error',
      'no-unsafe-finally': 'error',
      'no-unused-expressions': 'error',
      'no-unused-labels': 'error',
      'no-use-before-define': 'off',
      '@typescript-eslint/no-use-before-define': 'error',
      'no-var': 'error',
      'object-shorthand': 'error',
      'prefer-arrow/prefer-arrow-functions': 'error',
      'prefer-const': 'error',
      radix: 'error',
      'use-isnan': 'error',
    },
  },
  {
    files: ['node/**/*.ts'],

    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  {
    files: ['common/lib/**/*.ts', 'node/lib/**/*.ts'],

    rules: {
      'jsdoc/check-alignment': 'error',
      'jsdoc/check-indentation': 'error',
    },
  },
  {
    files: ['common/test/**/*.ts'],

    rules: {
      '@typescript-eslint/naming-convention': 'off',
      'import/no-extraneous-dependencies': 'off',
    },
  },
  {
    files: ['node/script/**/*.ts', 'node/test/**/*.ts', 'web/script/**/*.ts', 'web/test/**/*.ts'],

    rules: {
      '@typescript-eslint/naming-convention': 'off',
      '@typescript-eslint/no-empty-function': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-require-imports': 'off',
      '@typescript-eslint/no-var-requires': 'off',
      '@typescript-eslint/no-unnecessary-type-assertion': 'off',
      camelcase: 'off',
      'prefer-arrow/prefer-arrow-functions': 'off',
      'import/no-extraneous-dependencies': 'off',
      'import/no-unassigned-import': 'off',
      'import/no-internal-modules': 'off',
      'no-console': 'off',
      'no-empty': 'off',
      'no-unused-expressions': 'off',
    },
  },
  {
    files: ['web/lib/**/*.ts'],

    rules: {
      'no-underscore-dangle': [
        'error',
        {
          allow: [
            '_free',
            '_malloc',
            '_JsepGetNodeName',
            '_JsepOutput',
            '_OrtAddFreeDimensionOverride',
            '_OrtAddRunConfigEntry',
            '_OrtAddSessionConfigEntry',
            '_OrtAppendExecutionProvider',
            '_OrtBindInput',
            '_OrtBindOutput',
            '_OrtClearBoundOutputs',
            '_OrtCreateBinding',
            '_OrtCreateRunOptions',
            '_OrtCreateSession',
            '_OrtCreateSessionOptions',
            '_OrtCreateTensor',
            '_OrtEndProfiling',
            '_OrtFree',
            '_OrtGetInputName',
            '_OrtGetInputOutputCount',
            '_OrtGetInputOutputMetadata',
            '_OrtGetLastError',
            '_OrtGetOutputName',
            '_OrtGetTensorData',
            '_OrtInit',
            '_OrtReleaseBinding',
            '_OrtReleaseRunOptions',
            '_OrtReleaseSession',
            '_OrtReleaseSessionOptions',
            '_OrtReleaseTensor',
            '_OrtRun',
            '_OrtRunWithBinding',
          ],
        },
      ],
    },
  },
  {
    files: ['web/lib/onnxjs/**/*.ts'],

    rules: {
      '@typescript-eslint/no-empty-function': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-use-before-define': 'off',
      '@typescript-eslint/no-unnecessary-type-assertion': 'off',
      '@typescript-eslint/restrict-plus-operands': 'off',
      'import/no-internal-modules': 'off',
      'prefer-arrow/prefer-arrow-functions': 'off',
      'no-param-reassign': 'off',
      'no-underscore-dangle': 'off',
      'guard-for-in': 'off',
    },
  },
  {
    files: ['react_native/e2e/src/**/*.ts', 'react_native/e2e/src/**/*.tsx'],

    rules: {
      '@typescript-eslint/no-non-null-assertion': 'off',
      '@typescript-eslint/no-unnecessary-type-assertion': 'off',
      'unicorn/filename-case': 'off',
      'no-invalid-this': 'off',
      'no-console': 'off',
    },
  },
  {
    files: ['react_native/lib/**/*.ts'],

    rules: {
      '@typescript-eslint/naming-convention': 'off',
    },
  },
  {
    files: ['react_native/scripts/**/*.ts'],

    rules: {
      'import/no-extraneous-dependencies': 'off',
      'prefer-arrow/prefer-arrow-functions': 'off',
      'no-console': 'off',
    },
  },
  {
    files: ['scripts/**/*.ts'],

    rules: {
      'import/no-extraneous-dependencies': 'off',
      'no-console': 'off',
    },
  },
  {
    files: ['web/lib/**/3rd-party/**/*.ts'],

    rules: {
      'header/header': 'off',
      'unicorn/filename-case': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
    },
  },
];
