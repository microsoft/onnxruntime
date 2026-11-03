from build_qmoe_node_replay import replace_once


def transform(source):
    return replace_once(
        source,
        '            {"cache_key", pending_kernel_info.cache_key},\n'
        "        };",
        '            {"cache_key", pending_kernel_info.cache_key},\n'
        '            {"gpu_duration_ns", std::to_string(end_time - start_time)},\n'
        "        };")


def precise_duration_us(event):
    arguments = event.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("Missing explicit GPU duration argument map")
    raw = arguments.get("gpu_duration_ns")
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        raise ValueError("Missing or malformed explicit GPU nanosecond duration")
    nanoseconds = int(raw)
    if not 0 < nanoseconds <= 2**64 - 1:
        raise ValueError("Invalid GPU nanosecond duration")
    legacy = event.get("dur")
    if isinstance(legacy, bool) or legacy != (nanoseconds + 500) // 1000:
        raise ValueError("GPU nanoseconds disagree with unchanged rounded legacy duration")
    return nanoseconds / 1000.0
