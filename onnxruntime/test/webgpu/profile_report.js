// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* Browser client embedded by visualize_profile.mjs. No network requests or runtime dependencies. */
(async () => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ms = (us) => (us / 1000).toFixed(2);
  const percent = (part, total) => (total ? ((100 * part) / total).toFixed(1) : '0.0');
  const number = (value) => value.toLocaleString('en-US');
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  try {
    const bytes = Uint8Array.from(atob($('report-data').textContent.trim()), (char) => char.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const report = JSON.parse(await new Response(stream).text());
    const canvas = $('timeline');
    const overview = $('overview');
    const tooltip = $('tooltip');
    const labels = report.lanes;
    const { events, strings } = report;
    let active = report.runs.find((run) => !run.warmup) ?? report.runs[0];
    let filtered = [];
    let stage = 'all';
    let selected = null;
    let query = '';
    let matches = new Set();
    let view = [0, 1];
    let boxes = [];
    let overviewBoxes = [];
    let gesture = null;
    let dimensions = {};
    const left = 162;
    const top = 43;
    const rowHeight = 34;
    const plotHeight = labels.length * rowHeight;
    const label = (run) =>
      `Run ${run.index} · ${run.stage}${run.stage === 'decode' ? ` ${run.step}` : ''}${run.warmup ? ' · warmup' : ''}`;
    const bounds = () =>
      active
        ? [active.start - Math.max(1, active.duration * 0.008), active.start + Math.max(1, active.duration * 1.008)]
        : [0, 1];
    const inside = (event) => event[0] < view[1] && event[0] + event[1] >= view[0];

    $('capture-title').textContent = report.title;
    $('capture-subtitle').textContent = [
      report.capture.backend,
      report.capture.ortVersion ? `ORT ${report.capture.ortVersion}` : '',
      report.capture.promptTokens ? `${report.capture.promptTokens} prompt tokens` : '',
      `${number(report.gpuEvents)} GPU dispatches · ${report.runs.length} runs`,
    ]
      .filter(Boolean)
      .join('   /   ');
    const calibrated = report.alignment.length === 1 && report.alignment[0] === 'd3d12_clock_calibration';
    $('alignment').textContent = calibrated ? '✓ Calibrated CPU / GPU clock' : '≈ Approximate alignment';
    $('alignment').classList.toggle('approximate', !calibrated);
    if (!calibrated) {
      $('measurement-note').prepend('Approximate alignment cannot measure initial GPU queueing delay. ');
    }
    $('capture-details').textContent = JSON.stringify({ ...report.capture, alignment: report.alignment }, null, 2);
    if (report.syntheticRun)
      $('measurement-note').prepend(
        'No CPU model_run event was recorded; the selected interval covers GPU timestamps only. ',
      );
    $('warmup').checked = !report.runs.some((run) => !run.warmup);

    function setupCanvas(target) {
      const { width, height } = target.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      target.width = Math.round(width * ratio);
      target.height = Math.round(height * ratio);
      const context = target.getContext('2d');
      context.scale(ratio, ratio);
      return { context, width, height };
    }

    function updateMatches() {
      matches = new Set(
        (active?.indices ?? []).filter((id) => {
          const event = events[id];
          return (strings[event[3]] + ' ' + strings[event[4]]).toLowerCase().includes(query);
        }),
      );
      $('search-count').textContent = query ? `${number(matches.size)} matches · Enter to inspect` : '';
    }

    function setView(start, stop) {
      if (!active) return;
      const [min, max] = bounds();
      const width = Math.min(max - min, Math.max(1, stop - start));
      start = Math.max(min, Math.min(start, max - width));
      view = [start, start + width];
      draw();
    }

    function fitRun() {
      view = bounds();
      draw();
    }
    function zoom(factor, anchor = (view[0] + view[1]) / 2) {
      setView(anchor + (view[0] - anchor) * factor, anchor + (view[1] - anchor) * factor);
    }

    function draw() {
      const { context: ctx, width, height } = setupCanvas(canvas);
      dimensions = { width, height, plotWidth: Math.max(1, width - left - 18) };
      const scale = dimensions.plotWidth / (view[1] - view[0]);
      const x = (time) => left + (time - view[0]) * scale;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.textBaseline = 'middle';
      boxes = [];
      if (!active) {
        ctx.fillStyle = '#7a8c9a';
        ctx.fillText('No runs match this filter.', 24, 50);
        $('view-label').textContent = '';
        return;
      }
      const rawStep = (view[1] - view[0]) / Math.max(2, dimensions.plotWidth / 90);
      const magnitude = 10 ** Math.floor(Math.log10(Math.max(0.001, rawStep)));
      const step = [1, 2, 5, 10].find((item) => item * magnitude >= rawStep) * magnitude;
      for (let lane = 0; lane < labels.length; lane++) {
        const y = top + lane * rowHeight;
        ctx.fillStyle = lane === 10 ? '#f0faf6' : lane % 2 ? '#f9fbfc' : '#ffffff';
        ctx.fillRect(0, y, width, rowHeight);
        ctx.fillStyle = lane === 10 ? '#078363' : '#9aabb7';
        ctx.font = '9px "Segoe UI", sans-serif';
        ctx.fillText(labels[lane][1], 13, y + 10);
        ctx.fillStyle = '#345063';
        ctx.font = '11px "Segoe UI", sans-serif';
        ctx.fillText(labels[lane][0], 13, y + 24);
      }
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, 0, dimensions.plotWidth, height);
      ctx.clip();
      for (let tick = Math.ceil((view[0] - active.start) / step) * step; tick <= view[1] - active.start; tick += step) {
        const position = x(active.start + tick);
        ctx.strokeStyle = '#e8eef2';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(position, top - 4);
        ctx.lineTo(position, top + plotHeight);
        ctx.stroke();
        ctx.fillStyle = '#7a8c9a';
        ctx.font = '10px "Segoe UI", sans-serif';
        const text = `${(tick / 1000).toFixed(Math.max(0, 3 - Math.floor(Math.log10(step))))} ms`;
        ctx.fillText(text, position + 3, 25);
      }
      for (const id of active.indices) {
        const event = events[id];
        if (!inside(event)) continue;
        const lane = event[2];
        const begin = Math.max(left, x(event[0]));
        const end = Math.min(width - 18, x(event[0] + event[1]));
        const barWidth = Math.max(0.8, end - begin);
        const y = top + lane * rowHeight + 9;
        ctx.globalAlpha = !query || matches.has(id) ? 1 : 0.12;
        ctx.fillStyle = labels[lane][2];
        ctx.fillRect(begin, y, barWidth, 18);
        if (selected === id) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#142e3d';
          ctx.strokeRect(begin - 1, y - 2, barWidth + 2, 22);
        }
        if (barWidth > 65) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(begin + 4, y, barWidth - 8, 18);
          ctx.clip();
          ctx.fillStyle = '#ffffff';
          ctx.font = '10px "Segoe UI", sans-serif';
          ctx.fillText(strings[event[3]].split('&').at(-1).replace('_kernel_time', ''), begin + 5, y + 9);
          ctx.restore();
        }
        boxes.push({ id, x: begin, end: begin + Math.max(3, barWidth), y, bottom: y + 18 });
      }
      ctx.globalAlpha = 1;
      if (gesture && !gesture.pan) {
        ctx.fillStyle = '#0a9c7920';
        ctx.fillRect(
          Math.min(gesture.startX, gesture.lastX),
          top,
          Math.abs(gesture.lastX - gesture.startX),
          plotHeight,
        );
      }
      ctx.restore();
      ctx.fillStyle = '#8799a6';
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.fillText(`Run-relative axis · session origin ${ms(active.start)} ms`, left, height - 19);
      $('view-label').textContent = `${ms(view[1] - view[0])} ms visible`;
    }

    function inspect(id) {
      selected = id;
      $('event-times').replaceChildren();
      $('event-args').textContent = '';
      $('fit-event').disabled = id === null;
      if (id === null) {
        $('event-name').textContent = 'Inspect a moment';
        $('event-hint').textContent = 'Click any CPU span or GPU dispatch. Search also matches shapes and cache keys.';
      } else {
        const event = events[id];
        $('event-name').textContent = strings[event[3]];
        $('event-hint').textContent = `${labels[event[2]][1]} · ${labels[event[2]][0]}`;
        for (const [key, value] of [
          ['Duration', `${ms(event[1])} ms (${event[1]} µs)`],
          ['Start in run', `${ms(event[0] - active.start)} ms`],
          ['Session timestamp', `${ms(event[0])} ms`],
          ['Process / thread', `${event[5] ?? '—'} / ${event[6] ?? '—'}`],
        ]) {
          $('event-times').append(element('dt', key), element('dd', value));
        }
        $('event-args').textContent = JSON.stringify(JSON.parse(strings[event[4]]), null, 2);
      }
      draw();
    }

    function showPhases() {
      $('cpu-steps').replaceChildren();
      const ids = [5, 6, 7, 8, 9];
      for (const id of ids) {
        const row = element('div', undefined, 'cpu-step');
        const line = element('div', undefined, 'step-line');
        line.append(element('span', labels[id][0]), element('span', `${ms(active.cpu[id])} ms`, 'mono'));
        const track = element('div', undefined, 'step-track');
        const bar = element('i');
        bar.style.width = `${percent(active.cpu[id], active.duration)}%`;
        bar.style.background = labels[id][2];
        track.append(bar);
        row.append(line, track);
        $('cpu-steps').append(row);
      }
      $('programs').replaceChildren();
      const max = active.programs[0]?.us || 1;
      for (const program of active.programs.slice(0, 15)) {
        const row = element('div', undefined, 'program-row');
        row.tabIndex = 0;
        row.setAttribute('role', 'button');
        row.setAttribute('aria-label', `Highlight ${program.name}`);
        const name = element('div');
        name.append(element('strong', program.name));
        const track = element('div', undefined, 'program-track');
        const bar = element('i');
        bar.style.width = `${percent(program.us, max)}%`;
        track.append(bar);
        name.append(track);
        row.append(
          name,
          element('span', `${number(program.count)}×`, 'value muted'),
          element('span', `${ms(program.us)} ms`, 'value mono'),
        );
        const highlight = () => {
          $('search').value = program.name;
          query = program.name.toLowerCase();
          updateMatches();
          draw();
        };
        row.addEventListener('click', highlight);
        row.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            highlight();
          }
        });
        $('programs').append(row);
      }
    }

    function drawOverview() {
      const { context: ctx, width, height } = setupCanvas(overview);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      overviewBoxes = [];
      if (!filtered.length) return;
      const max = filtered.reduce((value, run) => Math.max(value, run.duration), 1);
      const slot = (width - 32) / filtered.length;
      for (let index = 0; index < filtered.length; index++) {
        const run = filtered[index];
        const x = 16 + index * slot;
        const w = Math.max(1, slot - Math.min(7, slot / 3));
        const total = (68 * run.duration) / max;
        const busy = (68 * run.busy) / max;
        ctx.fillStyle = run.index === active?.index ? '#fff1db' : '#f1f3f5';
        ctx.fillRect(x, 85 - total, w, total);
        ctx.fillStyle = run.index === active?.index ? '#08a67c' : '#91cdb9';
        ctx.fillRect(x, 85 - busy, w, busy);
        if (run.index === active?.index) {
          ctx.strokeStyle = '#23594c';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x - 2, 83 - total, w + 4, total + 4);
        }
        ctx.fillStyle = '#78909b';
        ctx.font = '9px "Segoe UI", sans-serif';
        if (slot > 20) ctx.fillText(String(run.index), x + w / 2 - 3, 100);
        overviewBoxes.push({ x, end: x + slot, run });
      }
      $('overview-label').textContent = `${filtered.length} runs · max ${ms(max)} ms`;
    }

    function selectRun(run) {
      active = run;
      tooltip.style.display = 'none';
      $('metric-run').textContent = ms(run.duration);
      $('metric-gpu').textContent = ms(run.busy);
      $('metric-gap').textContent = ms(run.gap);
      $('run-caption').textContent = `${label(run)}${run.inputTokens ? ` · ${run.inputTokens} input tokens` : ''}`;
      $('gpu-caption').textContent = `${percent(run.busy, run.duration)}% of run · compute interval union`;
      const baseline = Object.hasOwn(report.baseline, run.stage) ? report.baseline[run.stage] : undefined;
      $('metric-baseline').textContent = baseline === undefined ? '—' : baseline.toFixed(2);
      $('baseline-caption').textContent =
        baseline === undefined
          ? 'No separate unprofiled capture provided'
          : `${run.stage} · separate capture, no profiling`;
      $('run-select').value = String(run.index);
      for (const item of $('run-list').children)
        item.setAttribute('aria-current', item.dataset.run === String(run.index));
      $('previous').disabled = filtered.indexOf(run) <= 0;
      $('next').disabled = filtered.indexOf(run) >= filtered.length - 1;
      updateMatches();
      showPhases();
      view = bounds();
      inspect(null);
      drawOverview();
    }

    function filterRuns() {
      filtered = report.runs.filter(
        (run) => (stage === 'all' || run.stage === stage) && ($('warmup').checked || !run.warmup),
      );
      for (const id of ['run-select', 'search', 'zoom-in', 'zoom-out', 'reset', 'export-image'])
        $(id).disabled = !filtered.length;
      $('run-list').replaceChildren();
      $('run-select').replaceChildren();
      for (const run of filtered) {
        const button = element('button', undefined, 'run-item');
        button.dataset.run = String(run.index);
        button.setAttribute('aria-label', label(run));
        const line = element('span');
        line.append(
          element(
            'span',
            `${run.stage === 'prefill' ? 'Prefill' : run.stage === 'decode' ? `Decode ${run.step}` : 'Inference'}`,
          ),
          element('span', `${ms(run.duration)} ms`, 'mono'),
        );
        button.append(
          line,
          element('small', `Run ${run.index} · ${run.warmup ? 'warmup' : `iteration ${run.iteration ?? '—'}`}`),
        );
        const track = element('div', undefined, 'mini-track');
        const bar = element('i');
        bar.style.width = `${percent(run.busy, run.duration)}%`;
        track.append(bar);
        button.append(track);
        button.addEventListener('click', () => selectRun(run));
        $('run-list').append(button);
        const option = element('option', label(run));
        option.value = String(run.index);
        $('run-select').append(option);
      }
      if (!filtered.length) {
        active = null;
        tooltip.style.display = 'none';
        updateMatches();
        inspect(null);
        for (const id of ['metric-run', 'metric-gpu', 'metric-gap', 'metric-baseline']) $(id).textContent = '—';
        for (const id of ['run-caption', 'gpu-caption', 'baseline-caption', 'overview-label']) $(id).textContent = '';
        $('programs').replaceChildren();
        $('cpu-steps').replaceChildren();
        $('run-list').append(element('p', 'No runs match this filter.', 'empty'));
        $('previous').disabled = true;
        $('next').disabled = true;
        drawOverview();
        return;
      }
      selectRun(filtered.includes(active) ? active : filtered[0]);
    }

    const point = (event) => {
      const box = canvas.getBoundingClientRect();
      return { x: event.clientX - box.left, y: event.clientY - box.top };
    };
    const timeAt = (x) => view[0] + ((x - left) / dimensions.plotWidth) * (view[1] - view[0]);
    const hit = ({ x, y }) =>
      boxes
        .filter((box) => x >= box.x && x <= box.end && y >= box.y && y <= box.bottom)
        .sort((a, b) => events[a.id][1] - events[b.id][1])[0]?.id;
    canvas.addEventListener('pointerdown', (event) => {
      if (!active) return;
      const { x } = point(event);
      if (x < left) return;
      canvas.setPointerCapture(event.pointerId);
      gesture = { startX: x, lastX: x, view: [...view], pan: event.shiftKey };
      tooltip.style.display = 'none';
    });
    canvas.addEventListener('pointermove', (event) => {
      const p = point(event);
      if (gesture) {
        gesture.lastX = Math.max(left, Math.min(dimensions.width - 18, p.x));
        if (gesture.pan) {
          const delta = ((gesture.startX - gesture.lastX) / dimensions.plotWidth) * (gesture.view[1] - gesture.view[0]);
          setView(gesture.view[0] + delta, gesture.view[1] + delta);
        } else draw();
        return;
      }
      const id = hit(p);
      if (id === undefined) {
        tooltip.style.display = 'none';
        return;
      }
      const data = events[id];
      tooltip.replaceChildren(
        element('strong', strings[data[3]]),
        element('br'),
        element('span', `${labels[data[2]][1]} · ${ms(data[1])} ms · start +${ms(data[0] - active.start)} ms`),
      );
      tooltip.style.display = 'block';
      tooltip.style.left = `${Math.max(4, Math.min(event.clientX + 16, window.innerWidth - 400))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(event.clientY + 12, window.innerHeight - 100))}px`;
    });
    canvas.addEventListener('pointerup', (event) => {
      if (!gesture) return;
      const current = gesture;
      gesture = null;
      if (Math.abs(current.startX - current.lastX) < 4) inspect(hit(point(event)) ?? null);
      else if (!current.pan)
        setView(timeAt(Math.min(current.startX, current.lastX)), timeAt(Math.max(current.startX, current.lastX)));
      else draw();
    });
    canvas.addEventListener('pointercancel', () => {
      gesture = null;
      draw();
    });
    canvas.addEventListener('pointerleave', () => {
      tooltip.style.display = 'none';
    });
    canvas.addEventListener('dblclick', fitRun);
    canvas.addEventListener(
      'wheel',
      (event) => {
        if (!event.ctrlKey) return;
        event.preventDefault();
        zoom(event.deltaY > 0 ? 1.3 : 1 / 1.3, timeAt(point(event).x));
      },
      { passive: false },
    );
    overview.addEventListener('click', (event) => {
      const x = event.clientX - overview.getBoundingClientRect().left;
      const box = overviewBoxes.find((item) => x >= item.x && x <= item.end);
      if (box) selectRun(box.run);
    });
    $('run-select').addEventListener('change', () => {
      const run = filtered.find((item) => item.index === Number($('run-select').value));
      if (run) selectRun(run);
    });
    const nextRun = (offset) => {
      const run = filtered[filtered.indexOf(active) + offset];
      if (run) selectRun(run);
    };
    $('previous').addEventListener('click', () => nextRun(-1));
    $('next').addEventListener('click', () => nextRun(1));
    $('zoom-in').addEventListener('click', () => zoom(0.5));
    $('zoom-out').addEventListener('click', () => zoom(2));
    $('reset').addEventListener('click', fitRun);
    const fitEvent = () => {
      if (selected === null) return;
      const event = events[selected];
      const padding = Math.max(5, event[1] * 0.4);
      setView(event[0] - padding, event[0] + event[1] + padding);
    };
    $('fit-event').addEventListener('click', fitEvent);
    $('search').addEventListener('input', () => {
      query = $('search').value.trim().toLowerCase();
      updateMatches();
      draw();
    });
    $('search').addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || !matches.size) return;
      const ids = [...matches];
      const index = ids.indexOf(selected);
      inspect(ids[(index + 1) % ids.length]);
      fitEvent();
    });
    $('warmup').addEventListener('change', filterRuns);
    for (const tab of document.querySelectorAll('[data-stage]'))
      tab.addEventListener('click', () => {
        stage = tab.dataset.stage;
        for (const other of document.querySelectorAll('[data-stage]'))
          other.setAttribute('aria-pressed', String(other === tab));
        filterRuns();
      });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        $('search').value = '';
        query = '';
        updateMatches();
        inspect(null);
        tooltip.style.display = 'none';
        return;
      }
      if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName)) return;
      const actions = {
        ArrowLeft: () => nextRun(-1),
        ArrowRight: () => nextRun(1),
        '+': () => zoom(0.5),
        '=': () => zoom(0.5),
        '-': () => zoom(2),
        0: fitRun,
        '/': () => $('search').focus(),
      };
      if (actions[event.key]) {
        event.preventDefault();
        actions[event.key]();
      }
    });
    function download(blob, name) {
      const url = URL.createObjectURL(blob);
      const a = element('a');
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }
    $('export-image').addEventListener('click', () => {
      if (!active) return;
      const index = active.index;
      canvas.toBlob((blob) => {
        if (blob) download(blob, `webgpu-run-${index}.png`);
      });
    });
    $('export-summary').addEventListener('click', () => {
      const summary = {
        capture: report.capture,
        alignment: report.alignment,
        baseline: report.baseline,
        runs: report.runs.map(({ indices, ...run }) => run),
      };
      download(new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' }), 'webgpu-summary.json');
    });
    new ResizeObserver(() => {
      draw();
      drawOverview();
    }).observe(canvas.parentElement);
    new ResizeObserver(drawOverview).observe(overview.parentElement);
    filterRuns();
    $('loading').remove();
    document.documentElement.dataset.reportReady = 'true';
  } catch (error) {
    $('loading').textContent =
      `Unable to open report: ${error.message}. Use a recent Chrome, Edge or Firefox with gzip DecompressionStream support.`;
  }
})();
