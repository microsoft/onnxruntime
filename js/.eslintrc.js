module.exports = {
  root: true,
  ignorePatterns: ['**/*.js', 'node_modules/', 'types/'],
  env: { 'es6': true },
  parser: '@typescript-eslint/parser',
  parserOptions: { 'project': 'tsconfig.json', 'sourceType': 'module' },
  plugins: ['@typescript-eslint', 'prefer-arrow', 'import', 'jsdoc'],
  rules: {
    'import/no-extraneous-dependencies': ['error', { 'devDependencies': false }],
    'import/no-internal-modules': 'error',
    'import/no-unassigned-import': 'error',
    '@typescript-eslint/array-type': ['error', { 'default': 'array-simple' }],
    '@typescript-eslint/await-thenable': 'error',
    '@typescript-eslint/ban-types': [
      'error', {
        'types': {
          'Object': { 'message': 'Use {} instead.' },
          'String': { 'message': 'Use \'string\' instead.' },
          'Number': { 'message': 'Use \'number\' instead.' },
          'Boolean': { 'message': 'Use \'boolean\' instead.' }
        }
      }
    ],
    '@typescript-eslint/naming-convention': 'error',
    '@typescript-eslint/consistent-type-assertions': 'error',
    '@typescript-eslint/member-delimiter-style': [
      'error', {
        'multiline': { 'delimiter': 'semi', 'requireLast': true },
        'singleline': { 'delimiter': 'semi', 'requireLast': false }
      }
    ],
    '@typescript-eslint/no-empty-function': 'error',
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-for-in-array': 'error',
    '@typescript-eslint/no-inferrable-types': 'error',
    '@typescript-eslint/no-misused-new': 'error',
    '@typescript-eslint/no-namespace': ['error', { "allowDeclarations": true }],
    '@typescript-eslint/no-non-null-assertion': 'error',
    '@typescript-eslint/no-require-imports': 'error',
    '@typescript-eslint/no-unnecessary-type-assertion': 'error',
    '@typescript-eslint/no-unused-vars': ["error", { "argsIgnorePattern": "^_" }],
    '@typescript-eslint/promise-function-async': 'error',
    '@typescript-eslint/quotes': ['error', 'single'],
    '@typescript-eslint/restrict-plus-operands': 'error',
    '@typescript-eslint/semi': ['error', 'always'],
    '@typescript-eslint/triple-slash-reference':
      ['error', { 'path': 'always', 'types': 'prefer-import', 'lib': 'always' }],
    'arrow-body-style': 'error',
    'camelcase': 'error',
    'constructor-super': 'error',
    'curly': 'error',
    'default-case': 'error',
    'dot-notation': 'error',
    'eqeqeq': ['error', 'smart'],
    'guard-for-in': 'error',
    'id-match': 'error',
    'max-len': ['error', { 'code': 120 }],
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
    "@typescript-eslint/no-redeclare": ["error"],
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
    'radix': 'error',
    'use-isnan': 'error'
  },
  overrides: [{
    files: ['node/**/*.ts'],
    env: { 'es6': true, 'node': true }
  }, {
    files: ['common/lib/**/*.ts', 'node/lib/**/*.ts'],
    rules: {
      'jsdoc/check-alignment': 'error',
      'jsdoc/check-indentation': 'error',
      'jsdoc/newline-after-description': 'error',
    }
  }, {
    files: ['node/script/**/*.ts', 'node/test/**/*.ts', 'web/script/**/*.ts', 'web/test/**/*.ts'], rules: {
      '@typescript-eslint/naming-convention': 'off',
      '@typescript-eslint/no-empty-function': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-require-imports': 'off',
      '@typescript-eslint/no-var-requires': 'off',
      '@typescript-eslint/no-non-null-assertion': 'off',
      'camelcase': 'off',
      'prefer-arrow/prefer-arrow-functions': 'off',
      'import/no-extraneous-dependencies': 'off',
      'import/no-unassigned-import': 'off',
      'import/no-internal-modules': 'off',
      'no-console': 'off',
      'no-empty': 'off',
      'no-unused-expressions': 'off',
    }
  }, {
    files: ['web/lib/**/*.ts'], rules: {
      'no-underscore-dangle': 'off'
    }
  }, {
    files: ['web/lib/wasm/binding/**/*.ts'], rules: {
      '@typescript-eslint/naming-convention': 'off'
    }
  }],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/eslint-recommended',
    'plugin:@typescript-eslint/recommended',
  ],
};
