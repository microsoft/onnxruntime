from onnxruntime.transformers.benchmark_helper import measure_memory
import sys


def measure_fn():
    # Measure memory usage
    measure_memory(is_gpu=True, func=lambda: 3 * 6)

    sys.stdout.flush()


measure_fn()
