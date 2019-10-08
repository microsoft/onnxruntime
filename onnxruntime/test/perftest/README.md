# ONNXRuntime Performance Test

This tool provides the time for the ONNX Runtime with the specific execution provider to run a inference for a given model using the given test date. This tool can provide a reliable measurement for the inference latency usign ONNX Runtime on the HW platform. The options to use with the tool are listed below:

`onnxruntime_perf_test [options...] model_path result_file`

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
    Performance test uses the same input structure as *onnx_test_runner* tool. It requrires the directory trees as below:

    --ModelName
        --test_data_set_0
            --input0.pb
        --test_data_set_2
	        --input0.pb
        --model.onnx
    
The path of model.onnx needs to be provided as `<model_path>` argument.

__Sample output__ from the tool will look like something this:

	Total time cost:58.8053
	Total iterations:1000
	Average time cost:58.8053 ms
	Total run time:58.8102 s
	Min Latency is 0.0559777sec
	Max Latency is 0.0623472sec
	P50 Latency is 0.0587108sec
	P90 Latency is 0.0599845sec
	P95 Latency is 0.0605676sec
	P99 Latency is 0.0619517sec
	P999 Latency is 0.0623472se

