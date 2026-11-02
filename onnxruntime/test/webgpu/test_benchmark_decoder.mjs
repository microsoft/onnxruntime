// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'node:assert/strict';
import test from 'node:test';
import { median, renderReport, summarize } from './benchmark_decoder.mjs';

function capture(id, round, times = [2, 4], extra = {}) {
  return {
    id,
    label: id,
    round,
    capture: {
      trace: null,
      prompt_sha256: 'prompt',
      prompt_tokens: 128,
      decode_steps: 2,
      backend: 'D3D12',
      ort_version: 'test',
      warmup: 1,
      generated_token_ids: [[99], [1, 2, 3]],
      runs: [
        { warmup: true, stage: 'decode', wall_ms: 100, step_wall_ms: 100 },
        { warmup: false, stage: 'prefill', wall_ms: 20, step_wall_ms: 22 },
        ...times.map((time) => ({ warmup: false, stage: 'decode', wall_ms: time, step_wall_ms: time + 1 })),
      ],
      ...extra,
    },
  };
}

test('medians and round ranges exclude warmup and keep whole-step timing separate', () => {
  assert.equal(median([]), null);
  assert.equal(median([3, 1, 2]), 2);
  const [base] = summarize([capture('base', 0), capture('base', 1, [6, 8])]);
  assert.equal(base.metrics.decode.wall_ms, 5);
  assert.equal(base.metrics.decode.step_wall_ms, 6);
  assert.deepEqual(base.metrics.decode.wall_ms_rounds, [3, 7]);
  assert.equal(base.metrics.decode.samples, 4);
  assert.equal(base.sameTokens, true);
});

test('rejects profiling, mismatched workloads and invalid timings', () => {
  assert.throws(() => summarize([capture('a', 0, [1], { trace: 'profile.json' })]), /unprofiled/);
  for (const extra of [{ prompt_sha256: 'other' }, { prompt_tokens: 1 }, { decode_steps: 7 }, { backend: 'Metal' }]) {
    assert.throws(() => summarize([capture('a', 0), capture('b', 0, [1], extra)]), /Incompatible/);
  }
  assert.throws(() => summarize([capture('a', 0, [NaN])]), /Invalid/);
  assert.throws(() => summarize([capture('a', 0), capture('a', 1, [1], { static_kv: true })]), /Configuration changed/);
});

test('token comparisons do not substitute for numerical equivalence', () => {
  const rows = summarize([capture('a', 0), capture('b', 0, [1, 2], { generated_token_ids: [[99], [1, 8, 3]] })]);
  assert.equal(rows[1].sameTokens, false);
});

test('HTML embeds escaped data and only local timeline links', () => {
  const summary = summarize([capture('a', 0)]);
  const html = renderReport(summary, {
    title: '<script>title</script>',
    validation: '</script><img src=x>',
    timelines: [{ label: 'CPU + GPU', href: 'baseline/explorer.html' }],
  });
  assert.match(html, /&lt;script&gt;title/);
  assert.ok(!html.includes('</script><img'));
  assert.match(html, /\\u003c/);
  for (const href of ['../trace.html', 'https://other/trace.html', '/trace.html', 'javascript:trace.html']) {
    assert.throws(() => renderReport(summary, { timelines: [{ href }] }), /relative HTML/);
  }
});
