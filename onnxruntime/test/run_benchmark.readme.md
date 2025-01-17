# run_benchmark.py

`run_benchmark.py` is a helper script that runs a [Google Benchmark](https://github.com/google/benchmark) program
repeatedly until the measurements are within the desired
[coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) and then outputs the measurements.

It can be useful for obtaining measurements that are stable enough when repeated invocations of a benchmark program
show some measurement variance across runs.

Note that the script runs the benchmark program with specific options and parses specifically formatted output, so it
is only expected to work with Google Benchmark programs.

## Example usage

To run a benchmark program and get measurements for benchmark test(s) with a particular name:

```
python run_benchmark.py --program <path to benchmark program, e.g., onnxruntime_mlas_benchmark> --pattern <benchmark test name pattern>
```

For more detailed usage information, run it with the `--help` option.
