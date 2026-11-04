// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { BUDGET_BYTES, MAIN_REF, PILOT_REF, selectEvictions, listCaches, run } = require('./prune-macos-sccache.cjs');

function cache(id, size, variant = 'a', overrides = {}) {
  return {
    id,
    key: `sccache-v1-macOS-ARM64-${variant.repeat(64)}-${id.toString(16).padStart(40, '0')}`,
    ref: MAIN_REF,
    size_in_bytes: size,
    created_at: new Date(Date.UTC(2026, 10, 1, 0, id)).toISOString(),
    ...overrides,
  };
}

function pilot(id, size) {
  return cache(id, size, 'a', {
    ref: PILOT_REF,
    key: `macos-sccache-v2-arm64-arm64-Debug-${'a'.repeat(64)}-${'b'.repeat(40)}`,
  });
}

const ids = (plan) => plan.deletions.map((c) => c.id);
const pages = (entries) => JSON.stringify([{ actions_caches: entries }]);
const quiet = () => {};

test('scope excludes other OSes, vcpkg, unknown keys and all other refs', () => {
  const entries = [
    cache(1, 10, 'a', { key: 'vcpkg-macos-cache' }),
    cache(2, 10, 'a', { key: cache(2, 10).key.replace('macOS', 'Linux') }),
    cache(3, 10, 'a', { key: cache(3, 10).key.replace('macOS', 'Windows') }),
    cache(4, 10, 'a', { ref: 'refs/pull/42/merge' }),
    cache(5, 10, 'a', { ref: 'refs/heads/feature' }),
    cache(6, 10, 'a', { key: 'sccache-v1-macOS-unrecognized' }),
    { ...pilot(7, 10), ref: MAIN_REF },
    { ...pilot(8, 10), ref: 'refs/heads/feature' },
    cache(9, 10),
    pilot(10, 10),
  ];
  assert.deepEqual(ids(selectEvictions(entries, 0)), [10, 9]);
});

test('empty inventory, exact budget and under budget need no eviction', () => {
  assert.deepEqual(ids(selectEvictions([], 0)), []);
  assert.deepEqual(ids(selectEvictions([cache(1, BUDGET_BYTES)])), []);
  assert.deepEqual(ids(selectEvictions([cache(1, 1)])), []);
});

test('Intel main snapshots and both pilot key versions are included', () => {
  const intel = cache(1, 10, 'a', { key: cache(1, 10).key.replace('ARM64', 'X64') });
  const oldPilot = { ...pilot(2, 10), key: pilot(2, 10).key.replace('-v2-', '-v1-') };
  assert.deepEqual(ids(selectEvictions([intel, oldPilot, pilot(3, 10)], 0)), [2, 3, 1]);
});

test('aggregate budget evicts pilot then oldest generations, preserving newest per prefix', () => {
  const entries = [cache(1, 8), cache(2, 8, 'b'), cache(3, 8), cache(4, 8, 'b'), pilot(5, 8)];
  const plan = selectEvictions(entries.reverse(), 24);
  assert.deepEqual(ids(plan), [5, 1]);
  assert.equal(plan.totalBytes, 40);
  assert.equal(plan.remainingBytes, 24);
  assert.deepEqual(plan.evictedBaselines, []);
});

test('multiple generations and compatibility prefixes are independent', () => {
  const entries = Array.from({ length: 170 }, (_, i) => cache(i + 1, 1_000_000_000, (i % 10).toString(16)));
  const plan = selectEvictions(entries);
  assert.equal(plan.deletions.length, 140);
  assert.equal(plan.remainingBytes, BUDGET_BYTES);
  assert.deepEqual(ids(plan), Array.from({ length: 140 }, (_, i) => i + 1));
  assert.deepEqual(plan.evictedBaselines, []);
});

test('oversized protected set evicts oldest baselines explicitly', () => {
  const plan = selectEvictions([cache(1, 20), cache(2, 20, 'b')], 30);
  assert.deepEqual(ids(plan), [1]);
  assert.deepEqual(plan.evictedBaselines.map((c) => c.id), [1]);
  assert.deepEqual(ids(selectEvictions([cache(1, 31)], 30)), [1]);
});

test('equal creation times break ties by ID, not inventory order or access time', () => {
  const a = cache(1, 10);
  const b = cache(2, 10, 'a', { created_at: a.created_at, last_accessed_at: '2000-01-01T00:00:00Z' });
  assert.deepEqual(ids(selectEvictions([b, a], 10)), [1]);
});

test('invalid budgets and scoped metadata fail closed', () => {
  for (const budget of [-1, NaN, 1.5]) {
    assert.throws(() => selectEvictions([], budget));
  }
  for (const overrides of [{ id: -1 }, { size_in_bytes: -1 }, { created_at: 'invalid' }]) {
    assert.throws(() => selectEvictions([cache(1, 1, 'a', overrides)]));
  }
});

test('both namespaces use complete pagination and duplicate IDs are counted once', () => {
  const calls = [];
  const entries = listCaches('owner/repo', (args) => {
    calls.push(args);
    return JSON.stringify([{ actions_caches: [cache(1, 1)] }, { actions_caches: [cache(2, 1)] }]);
  });
  assert.equal(entries.length, 2);
  assert.equal(calls.length, 2);
  assert.ok(calls.every((args) => args.includes('--paginate') && args.includes('--slurp')));
  assert.ok(calls[0].at(-1).includes('per_page=100&key=sccache-v1-macOS-'));
  assert.ok(calls[1].at(-1).includes('per_page=100&key=macos-sccache-v'));
});

test('dry run never deletes and reports baseline loss', () => {
  const messages = [];
  run({
    repository: 'owner/repo',
    api: (args) => {
      assert.ok(!args.includes('DELETE'));
      return pages([cache(1, BUDGET_BYTES + 1)]);
    },
    log: (message) => messages.push(message),
  });
  assert.ok(messages.some((message) => message.includes('::warning::')));
  assert.ok(messages.some((message) => message.includes('Would delete cache 1')));
});

test('apply deletes only exact scoped IDs and rechecks observed aggregate', () => {
  const calls = [];
  let lists = 0;
  run({
    repository: 'owner/repo',
    apply: true,
    api: (args) => {
      calls.push(args);
      if (args.includes('DELETE')) {
        return '';
      }
      return pages(++lists <= 2 ? [cache(1, BUDGET_BYTES), cache(2, 1)] : [cache(2, 1)]);
    },
    log: quiet,
  });
  assert.deepEqual(calls.filter((args) => args.includes('DELETE')), [
    ['--method', 'DELETE', 'repos/owner/repo/actions/caches/1'],
  ]);
  assert.equal(lists, 4);
});

test('listing, malformed response and deletion failures propagate', () => {
  assert.throws(() => run({ repository: 'owner/repo', api: () => { throw new Error('403'); }, log: quiet }), /403/);
  for (const response of ['bad JSON', '[]', '[{}]']) {
    assert.throws(() => listCaches('owner/repo', () => response));
  }
  assert.throws(() => run({
    repository: 'owner/repo', apply: true, log: quiet,
    api: (args) => {
      if (args.includes('DELETE')) {
        throw new Error('503');
      }
      return pages([cache(1, BUDGET_BYTES + 1)]);
    },
  }), /503/);
});

test('failure listing the second namespace prevents all deletion', () => {
  let calls = 0;
  assert.throws(() => run({
    repository: 'owner/repo', apply: true, log: quiet,
    api: (args) => {
      assert.ok(!args.includes('DELETE'));
      if (++calls === 2) {
        throw new Error('Second namespace failed');
      }
      return pages([cache(1, BUDGET_BYTES + 1)]);
    },
  }), /Second namespace failed/);
});

test('concurrent uploads exceeding budget are reported as failure after deletion', () => {
  assert.throws(() => run({
    repository: 'owner/repo', apply: true, log: quiet,
    api: (args) => args.includes('DELETE') ? '' : pages([cache(1, BUDGET_BYTES + 1)]),
  }), /remains over budget/);
});
