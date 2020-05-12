"""
Example python code for creating a model with a single operator and performance testing it with various
input combinations.
"""

import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
import time
import timeit
import onnxruntime as rt

# if you copy this script elsewhere you may need to add the tools\python dir to the sys.path for this
# import to work.
# e.g. sys.path.append(r'<path to onnxruntime source>\tools\python')
import ort_test_dir_utils

# make input deterministic
np.random.seed(123)


#
# Example code to create a model with just the operator to test. Adjust as necessary for what you want to test.
#
def create_model(model_name):
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(op_type="TopK", inputs=['X', 'K'], outputs=['Values', 'Indices'], name='topk',
                             # attributes are also key-value pairs using the attribute name and appropriate type
                             largest=1),
        ],
        name='test-model',
        inputs=[
            # create inputs with symbolic dims so we can use any input sizes
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ['batch', 'items']),
            helper.make_tensor_value_info("K", TensorProto.INT64, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("Values", TensorProto.FLOAT, ['batch', 'k']),
            helper.make_tensor_value_info("Indices", TensorProto.INT64, ['batch', 'k']),
        ],
        initializer=[
        ]
    )

    model = helper.make_model(graph_def, opset_imports=[helper.make_operatorsetid("", 11)])
    onnx.checker.check_model(model)

    onnx.save_model(model, model_name)


#
# Example code to create random input. Adjust as necessary for the input your model requires
#
def create_test_input(n, num_items, k):
    x = np.random.randn(n, num_items).astype(np.float32)
    k_in = np.asarray([k]).astype(np.int64)
    inputs = {'X': x, 'K': k_in}

    return inputs


#
# Example code that tests various combinations of input sizes.
#
def run_perf_tests(model_path, num_threads=1):

    so = rt.SessionOptions()
    so.intra_op_num_threads = num_threads
    sess = rt.InferenceSession(model_path, sess_options=so)

    batches = [10, 25, 50]
    batch_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    k_vals = [1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 128]

    # exploit scope to access variables from below for each iteration
    def run_test():
        num_seconds = 1 * 1000 * 1000 * 1000  # seconds in ns
        iters = 0
        total = 0
        total_iters = 0

        # For a simple model execution can be faster than time.time_ns() updates. Due to this we want to estimate
        # a number of iterations per measurement.
        # Estimate based on iterations in 5ms, but note that 5ms includes all the time_ns calls
        # which are excluded in the real measurement. The actual time that many iterations
        # takes will be much lower if the individual execution time is very small.
        start = time.time_ns()
        while time.time_ns() - start < 5 * 1000 * 1000:  # 5 ms
            sess.run(None, inputs)
            iters += 1

        # run the model and measure time after 'iters' calls
        while total < num_seconds:
            start = time.time_ns()
            for i in range(iters):
                # ignore the outputs as we're not validating them in a performance test
                sess.run(None, inputs)
            end = time.time_ns()
            assert (end - start > 0)
            total += end - start
            total_iters += iters

        # Adjust the output you want as needed
        print(f'n={n},items={num_items},k={k},avg:{total / total_iters:.4f}')

    # combine the various input parameters and create input for each test
    for n in batches:
        for num_items in batch_size:
            for k in k_vals:
                if k < num_items:
                    # adjust as necessary for the inputs your model requires
                    inputs = create_test_input(n, num_items, k)

                    # use timeit to disable gc etc. but let each test measure total time and average time
                    # as multiple iterations may be required between each measurement
                    timeit.timeit(lambda: run_test(), number=1)


#
# example for creating a test directory for use with onnx_test_runner or onnxruntime_perf_test
# so that the model can be easily run directly or from a debugger.
#
def create_example_test_directory():

    # fill in the inputs that we want to use specific values for
    input_data = {}
    input_data['K'] = np.asarray([64]).astype(np.int64)

    # provide symbolic dim values as needed
    symbolic_dim_values = {'batch': 25, 'items': 256}

    # create the directory. random input will be created for any missing inputs.
    # the model will be run and the output will be saved as expected output for future runs
    ort_test_dir_utils.create_test_dir('topk.onnx', 'PerfTests', 'test1', input_data, symbolic_dim_values)


# this will create the model file in the current directory
create_model('topk.onnx')

# this will create a test directory that can be used with onnx_test_runner or onnxruntime_perf_test
create_example_test_directory()

# this can loop over various combinations of input, using the specified number of threads
run_perf_tests('topk.onnx', 1)
