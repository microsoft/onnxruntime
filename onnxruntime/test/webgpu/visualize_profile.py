# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Render an ORT WebGPU trace as a self-contained, aligned CPU/GPU HTML report.

Requires plotly. Timestamps are preserved; approximate alignment is identified.
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go


def merged_duration(intervals):
    """Return the duration of the interval union, without double-counting overlap."""
    end = float("-inf")
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def lane(event):
    args = event.get("args", {})
    if "clock_alignment" in args:
        return "GPU · compute"
    name = event["name"]
    if name.startswith("WebGPU::"):
        return "CPU · " + name.removeprefix("WebGPU::")
    if name == "model_run":
        return "CPU · model run"
    if name == "SequentialExecutor::Execute":
        return "CPU · graph execution"
    if event.get("cat") == "Node" and name.endswith("_kernel_time"):
        if args.get("op_name", "").startswith("Memcpy"):
            return "CPU · tensor copies"
        if args.get("provider") == "CPUExecutionProvider":
            return "CPU · fallback operators"
        return "CPU · operator preparation"
    return None


def summarize(events, metadata):
    gpu = [event for event in events if "clock_alignment" in event.get("args", {})]
    if not gpu:
        raise ValueError("No WebGPU timestamp events found. Enable profiling and check timestamp-query support.")
    alignment = sorted({event["args"]["clock_alignment"] for event in gpu})
    runs = sorted((event for event in events if event.get("name") == "model_run"), key=lambda event: event["ts"])
    captures = metadata.get("runs", [])
    if captures and len(captures) != len(runs):
        raise ValueError("Capture metadata run count does not match the trace.")
    rows = []
    for index, run in enumerate(runs):
        start, stop = run["ts"], run["ts"] + run["dur"]
        intervals = [
            (max(start, event["ts"]), min(stop, event["ts"] + event["dur"]))
            for event in gpu
            if event["ts"] < stop and event["ts"] + event["dur"] > start
        ]
        busy = merged_duration(intervals)
        cpu_steps = defaultdict(float)
        for event in events:
            if event.get("args", {}).get("execution_side") == "cpu":
                overlap = min(stop, event["ts"] + event["dur"]) - max(start, event["ts"])
                if overlap > 0:
                    cpu_steps[event["name"]] += overlap / 1000
        label = captures[index] if captures else {}
        rows.append(
            {
                "index": index,
                "start_us": start,
                "duration_ms": run["dur"] / 1000,
                "gpu_busy_ms": busy / 1000,
                "gpu_gap_ms": max(0, run["dur"] - busy) / 1000,
                "cpu_steps_ms": dict(cpu_steps),
                **label,
            }
        )
    by_program = defaultdict(float)
    for event in gpu:
        by_program[event["name"].rsplit("&", 1)[-1]] += event["dur"] / 1000
    return {
        "alignment": alignment,
        "gpu_events": len(gpu),
        "runs": rows,
        "gpu_program_ms": dict(sorted(by_program.items(), key=lambda item: item[1], reverse=True)),
    }


def bar_data(events, group, run=None):
    if run:
        start = run["start_us"]
        stop = start + run["duration_ms"] * 1000
        events = [event for event in events if event["ts"] < stop and event["ts"] + event["dur"] > start]
    fields = {"op_name", "provider", "clock_alignment", "shapes", "execution_side"}
    details = []
    for event in events:
        args = {key: value for key, value in event.get("args", {}).items() if key in fields}
        args.update({"pid": event.get("pid"), "tid": event.get("tid")})
        description = "<br>".join(f"{key}: {html.escape(str(value)[:240])}" for key, value in args.items())
        details.append([html.escape(event["name"]), description])
    return {
        "y": [group] * len(events),
        "base": [event["ts"] / 1000 for event in events],
        "x": [event["dur"] / 1000 for event in events],
        "customdata": details,
    }


def render(trace_path, output, metadata):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    events = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) >= 0]
    summary = summarize(events, metadata)
    calibrated = summary["alignment"] == ["d3d12_clock_calibration"]
    note = (
        "D3D12 clock calibration: CPU and GPU timestamps share the ORT origin."
        if calibrated
        else "APPROXIMATE first-submit alignment: initial GPU queueing delay cannot be measured."
    )
    groups = defaultdict(list)
    for event in events:
        if group := lane(event):
            groups[group].append(event)
    order = [
        "CPU · model run",
        "CPU · graph execution",
        "CPU · operator preparation",
        "CPU · fallback operators",
        "CPU · tensor copies",
        "CPU · Encode",
        "CPU · Submit",
        "CPU · Wait",
        "CPU · StartProfiling",
        "CPU · ProfileReadback",
        "GPU · compute",
    ]
    colors = [
        "#64748b",
        "#94a3b8",
        "#2563eb",
        "#e11d48",
        "#0891b2",
        "#7c3aed",
        "#db2777",
        "#f59e0b",
        "#a16207",
        "#ea580c",
        "#16a34a",
    ]
    measured = [run for run in summary["runs"] if not run.get("warmup")]
    initial_run = measured[0] if measured else None
    figure = go.Figure()
    for group, color in zip(order, colors, strict=True):
        figure.add_trace(
            go.Bar(
                name=group,
                orientation="h",
                **bar_data(groups[group], group, initial_run),
                marker_color=color,
                hovertemplate="%{customdata[0]}<br>start %{base:.3f} ms · duration %{x:.3f} ms<br>%{customdata[1]}<extra></extra>",
            )
        )
    buttons = []
    for run in summary["runs"]:
        label = f"Run {run['index']} · {run.get('stage', 'inference')} {run.get('step', '')}"
        if run.get("warmup"):
            label += " · warmup"
        begin = run["start_us"] / 1000
        data = [bar_data(groups[group], group, run) for group in order]
        updates = {key: [item[key] for item in data] for key in ("x", "y", "base", "customdata")}
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [updates, {"xaxis.range": [begin - 0.2, begin + run["duration_ms"] + 0.2]}],
            }
        )
    initial_axis = {"title": "Milliseconds since ORT profiling started", "rangeslider": {"visible": True}}
    if measured:
        begin = measured[0]["start_us"] / 1000
        initial_axis["range"] = [begin - 0.2, begin + measured[0]["duration_ms"] + 0.2]
    figure.update_layout(
        title="CPU major steps and GPU kernels · one shared time axis",
        barmode="overlay",
        height=620,
        template="plotly_white",
        showlegend=False,
        margin={"t": 105, "l": 195, "r": 25, "b": 70},
        xaxis=initial_axis,
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(order))},
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.15, "active": initial_run["index"] if initial_run else 0}],
        dragmode="zoom",
    )
    overview = go.Figure()
    labels = [f"{run['index']}: {run.get('stage', 'run')} {run.get('step', '')}" for run in measured]
    overview.add_trace(
        go.Bar(
            name="GPU kernels (interval union)",
            x=labels,
            y=[run["gpu_busy_ms"] for run in measured],
            marker_color="#16a34a",
        )
    )
    overview.add_trace(
        go.Bar(
            name="Run time outside GPU kernels",
            x=labels,
            y=[run["gpu_gap_ms"] for run in measured],
            marker_color="#f59e0b",
        )
    )
    overview.update_layout(
        title="Measured runs · GPU work and gaps",
        barmode="stack",
        template="plotly_white",
        height=330,
        yaxis_title="Milliseconds",
    )
    programs = list(summary["gpu_program_ms"].items())[:15]
    table = "".join(f"<tr><td>{html.escape(name)}</td><td>{duration:.3f}</td></tr>" for name, duration in programs)
    counts = Counter(run.get("stage", "inference") for run in measured)
    description = html.escape(metadata.get("workload", str(trace_path)))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>WebGPU profile</title>
<style>body{{font:16px system-ui;margin:32px;color:#172033}}h1{{font-size:28px}}p{{max-width:1100px;line-height:1.5}}
table{{border-collapse:collapse}}td,th{{padding:7px 16px;border-bottom:1px solid #e2e8f0;text-align:left}}</style></head>
<body><h1>WebGPU CPU / GPU profile</h1><p><b>{html.escape(note)}</b></p><p>{description}</p>
<p>{summary["gpu_events"]} GPU dispatches; measured stages: {html.escape(str(dict(counts)))}.
Select a run, drag to zoom, and hover for event details. Double-click resets the view.</p>
{figure.to_html(full_html=False, include_plotlyjs=True)}
<p>CPU operator spans include host preparation and may overlap GPU execution. Wait/readback spans can overlap GPU
work and are nested inside other CPU spans; do not add their durations. Gaps include CPU work, transfers and scheduling;
they are not automatically CPU bottlenecks. Profiling introduces timestamp readback and synchronization overhead.
Only compute dispatches have GPU timestamps; transfers are not counted as GPU compute busy time.
Hover arguments are abbreviated; full shapes and cache keys remain in the original JSON trace.</p>
{overview.to_html(full_html=False, include_plotlyjs=False)}
<h2>Largest GPU program totals (all runs, including warmup)</h2><table><tr><th>Program</th><th>GPU ms</th></tr>{table}</table>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    render(args.trace, args.output, metadata)


if __name__ == "__main__":
    main()
