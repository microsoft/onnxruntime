import math


def verify_two_requests(raw, references, context):
    samples = raw["samples"]
    if len(samples) != 2 or [sample["warmup"] for sample in samples] != [True, False] or len(references) < 2:
        raise RuntimeError("CPU diagnostic requires one warmup and one measured request")
    for sample, reference in zip(samples, references[:2], strict=True):
        if (sample["context"] != context or sample["input_tokens"] != context or
                sample["prompt_sha256"] != reference["prompt_sha256"] or
                sample["output_tokens"] != 256 or len(sample["output_token_ids"]) != 256 or
                sample["output_token_ids"] != reference["output_token_ids"][:256]):
            raise RuntimeError("Diagnostic prompt or complete 256-token prefix differs from its frozen reference")
        if not sample["process_resource_usage"] or any(
                usage["major_page_faults"] != 0 for usage in sample["process_resource_usage"].values()):
            raise RuntimeError("Diagnostic incurred major page faults or lacks resource counters")
        if any(not math.isfinite(sample[key]) or sample[key] <= 0
               for key in ("decode_tps", "decode_seconds", "request_seconds", "ttft_seconds", "prefill_seconds")):
            raise RuntimeError("Nonpositive or nonfinite diagnostic timing")
        if (not math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12) or
                sample["prefill_timing_scope"] != "append_tokens_through_first_token_selection"):
            raise RuntimeError("Wrong decode denominator or prefill boundary in the CPU diagnostic")
