# ONNXRuntime Performance Test

onnxruntime_perf_test [options...] model_path result_file
Options:
        -m [test_mode]: Specifies the test mode. Value coulde be 'duration' or 'times'.
                Provide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. Default:'duration'.
        -e [cpu|cuda|mkldnn|tensorrt]: Specifies the provider 'cpu','cuda','mkldnn' or 'tensorrt'. Default:'cpu'.
        -r [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.
        -t [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.
        -p [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.
        -s: Show statistics result, like P75, P90.
        -v: Show verbose information.
        -x: Use parallel executor, default (without -x): sequential executor.
        -h: help

Model path and input data dependency:
    Performance test uses the same input structure as onnx_test_runner. It requrires the directory trees as below:

    --ModelName
        --test_data_set_0
            --input0.pb
        --test_data_set_2
	        --input0.pb
        --model.onnx
    The path of model.onnx needs to be provided as <model_path> argument.
