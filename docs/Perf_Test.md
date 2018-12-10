First things first: It's easy to gather some numbers in your system and publish them, it's hard to make them convincing. Please do perf test seriously and make your experiment reproducible. 

# Preparing

## Disable CPU frequency scaling

On most systems, CPU frequency is scaled automatically depending on the system load. Please disable it.

1. Append 'intel\_idle.max\_cstate=0 idle=poll' to your kernel command line args at booting.
2. sudo cpupower frequency-set -g performance
3. Use powertop to check if all the cpus are always in C0 state.

## Shutdown your background process


## Set scheduling policy to FIFO
man chrt(1)

## Get a tool for running Student's t-test 
Whenever doing a performance comparison, always use statistical method to check your conclusion. Student's t-test is a good candidate.

# Get test data
You can get sample data from https://github.com/onnx/models, pick a model, download the model file and the test data with it.


# Build onnxruntime
Add '-march=native -mtune=native' to your CFLAGS and CXXFLAGS

# Run perf test




