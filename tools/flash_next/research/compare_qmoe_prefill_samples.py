import math

from qualify_row4_cpu_locality import compare_samples


def compare_prefill_samples(candidate, before, after):
    result = compare_samples(candidate, before, after)
    ratios = [selected["ttft_seconds"] / math.sqrt(first["ttft_seconds"] * last["ttft_seconds"])
              for selected, first, last in zip(candidate, before, after)]
    positive = sum(ratio < 1 for ratio in ratios)
    decode_safe = result["matched_request_decode_ratio"] >= 0.99
    result.update(ttft_request_ratios=ratios, positive_ttft_requests=positive,
                  decode_regression_screen_passed=decode_safe, decode_regression_floor=0.99,
                  supported=positive == 3 and result["matched_request_ttft_ratio"] < 1 and decode_safe)
    return result
