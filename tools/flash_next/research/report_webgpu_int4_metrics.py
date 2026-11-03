import math


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Percentile requires samples and a quantile in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(raw, guard, context, requests):
    if raw["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0:
        raise ValueError("Incomplete benchmark or guard")
    samples = [sample for sample in raw["samples"] if not sample["warmup"]]
    if len(samples) != requests or sorted(sample["repetition"] for sample in samples) != list(range(requests)):
        raise ValueError("Measured request count or repetition identity mismatch")
    for sample in samples:
        if sample["context"] != context or sample["input_tokens"] != context:
            raise ValueError("Context mismatch")
        for field in ("prefill_seconds", "ttft_seconds", "decode_seconds", "request_seconds"):
            if not math.isfinite(sample[field]) or sample[field] <= 0:
                raise ValueError(f"Invalid {field}")
        if not sample["prefill_seconds"] <= sample["ttft_seconds"] <= sample["request_seconds"]:
            raise ValueError("Invalid phase timing boundaries")
        if sample["output_tokens"] != len(sample["output_token_ids"]) or sample["output_tokens"] < 2:
            raise ValueError("Incomplete output")
        if not math.isclose(sample["decode_tps"], (sample["output_tokens"] - 1) / sample["decode_seconds"],
                            rel_tol=1e-12):
            raise ValueError("Decode throughput formula mismatch")
        if not math.isclose(sample["request_seconds"], sample["ttft_seconds"] + sample["decode_seconds"],
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("Request duration decomposition mismatch")
    if guard["gpu_memory_samples"] < 2 or guard["peak_gpu_used_bytes"] <= 0:
        raise ValueError("Missing sampled peak GPU measurement")
    return dict(context={2048: "2K", 8192: "8K", 32768: "32K"}[context],
        requests_done=f"{len(samples)}/{requests}",
        ttft_p50_seconds=percentile([sample["ttft_seconds"] for sample in samples], 0.5),
        ttft_p95_seconds=percentile([sample["ttft_seconds"] for sample in samples], 0.95),
        prefill_tps=sum(sample["input_tokens"] for sample in samples) /
                    sum(sample["prefill_seconds"] for sample in samples),
        decode_tps=sum(sample["output_tokens"] - 1 for sample in samples) /
                   sum(sample["decode_seconds"] for sample in samples),
        peak_gpu_gib=guard["peak_gpu_used_bytes"] / 1024**3,
        gpu_memory_sampling_interval_seconds=guard["gpu_memory_sampling_interval_seconds"],
        gpu_memory_scope=guard["gpu_memory_scope"])


def table(rows):
    header = ("| Context | Requests done | TTFT P50 (s) | TTFT P95 (s) | Prefill (tok/s) | "
              "Decode (tok/s) | Peak GPU (GiB) |\n"
              "|---|---:|---:|---:|---:|---:|---:|")
    body = [f"| {row['context']} | {row['requests_done']} | {row['ttft_p50_seconds']:.3f} | "
            f"{row['ttft_p95_seconds']:.3f} | {row['prefill_tps']:.3f} | "
            f"{row['decode_tps']:.3f} | {row['peak_gpu_gib']:.3f} |" for row in rows]
    return "\n".join([header, *body])
