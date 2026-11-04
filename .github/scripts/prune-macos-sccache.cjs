// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const { execFileSync } = require('node:child_process');

const BUDGET_BYTES = 30_000_000_000;
const MAIN_REF = 'refs/heads/main';
const PILOT_REF = 'refs/heads/eserscor-dev/erscor/2026/10/30-mac-compiler-cachi';
const MAIN_KEY = /^(sccache-v1-macOS-(?:ARM64|X64)-[a-f0-9]{64}-)[a-f0-9]{40}$/;
const PILOT_KEY = /^macos-sccache-v[12]-.+-[a-f0-9]{64}-[a-f0-9]{40}$/;

function compatibility(cache) {
  if (cache.ref === MAIN_REF) {
    return MAIN_KEY.exec(cache.key)?.[1];
  }
  return undefined;
}

function inScope(cache) {
  return Boolean(compatibility(cache) || (cache.ref === PILOT_REF && PILOT_KEY.test(cache.key)));
}

function selectEvictions(inventory, budget = BUDGET_BYTES) {
  if (!Number.isSafeInteger(budget) || budget < 0) {
    throw new Error('Cache budget must be a nonnegative safe integer');
  }
  const caches = inventory.filter(inScope);
  for (const cache of caches) {
    if (
      !Number.isSafeInteger(cache.id) || cache.id <= 0 ||
      !Number.isSafeInteger(cache.size_in_bytes) || cache.size_in_bytes < 0 ||
      !Number.isFinite(Date.parse(cache.created_at))
    ) {
      throw new Error(`Invalid cache metadata for ${cache.key}`);
    }
  }
  caches.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.id - a.id);
  const baselines = new Set();
  const prefixes = new Set();
  for (const cache of caches) {
    const prefix = compatibility(cache);
    if (prefix && !prefixes.has(prefix)) {
      prefixes.add(prefix);
      baselines.add(cache.id);
    }
  }
  // Retire pilot snapshots first, then old main generations, and only then main baselines.
  const priority = (cache) => baselines.has(cache.id) ? 2 : compatibility(cache) ? 1 : 0;
  const candidates = [...caches].sort((a, b) =>
    priority(a) - priority(b) || Date.parse(a.created_at) - Date.parse(b.created_at) || a.id - b.id);
  const totalBytes = caches.reduce((sum, cache) => sum + cache.size_in_bytes, 0);
  if (!Number.isSafeInteger(totalBytes)) {
    throw new Error('Aggregate cache size exceeds safe integer range');
  }
  let remainingBytes = totalBytes;
  const deletions = [];
  for (const cache of candidates) {
    if (remainingBytes <= budget) {
      break;
    }
    deletions.push(cache);
    remainingBytes -= cache.size_in_bytes;
  }
  return { totalBytes, remainingBytes, deletions, evictedBaselines: deletions.filter((c) => baselines.has(c.id)) };
}

function ghApi(args) {
  return execFileSync('gh', ['api', ...args], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

function listCaches(repository, api) {
  const caches = new Map();
  for (const key of ['sccache-v1-macOS-', 'macos-sccache-v']) {
    const pages = JSON.parse(api([
      '--paginate', '--slurp', `repos/${repository}/actions/caches?per_page=100&key=${key}`,
    ]));
    if (!Array.isArray(pages) || pages.length === 0) {
      throw new Error('Invalid cache pagination response');
    }
    for (const page of pages) {
      if (!Array.isArray(page.actions_caches)) {
        throw new Error('Missing actions_caches in API response');
      }
      for (const cache of page.actions_caches) {
        caches.set(cache.id, cache);
      }
    }
  }
  return [...caches.values()];
}

function run({ repository, apply = false, api = ghApi, log = console.log }) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) {
    throw new Error('Expected repository in owner/name format');
  }
  const plan = selectEvictions(listCaches(repository, api));
  log(`${apply ? 'Apply' : 'Dry run'}: ${plan.totalBytes} bytes; budget ${BUDGET_BYTES}; ` +
    `${plan.deletions.length} evictions; planned remainder ${plan.remainingBytes} bytes`);
  if (plan.evictedBaselines.length > 0) {
    log(`::warning::Budget requires evicting ${plan.evictedBaselines.length} newest compatibility baselines`);
  }
  for (const cache of plan.deletions) {
    // IDs are immutable; scope is checked again at the destructive boundary.
    if (!inScope(cache) || !Number.isSafeInteger(cache.id) || cache.id <= 0) {
      throw new Error('Refusing to delete an out-of-scope cache');
    }
    log(`${apply ? 'Deleting' : 'Would delete'} cache ${cache.id}: ${cache.ref} ${cache.key} (${cache.size_in_bytes} bytes)`);
    if (apply) {
      api(['--method', 'DELETE', `repos/${repository}/actions/caches/${cache.id}`]);
    }
  }
  if (apply) {
    const observed = selectEvictions(listCaches(repository, api));
    log(`Observed storage after cleanup: ${observed.totalBytes} bytes`);
    if (observed.totalBytes > BUDGET_BYTES) {
      throw new Error('macOS sccache remains over budget (possibly concurrent uploads); next maintenance run will retry');
    }
  }
  return plan;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length === 1 && args[0] !== '--apply')) {
    throw new Error('Usage: node prune-macos-sccache.cjs [--apply] (default: dry run)');
  }
  run({ repository: process.env.GITHUB_REPOSITORY, apply: args[0] === '--apply' });
}

module.exports = { BUDGET_BYTES, MAIN_REF, PILOT_REF, selectEvictions, listCaches, run };
