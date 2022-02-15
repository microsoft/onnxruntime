# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import timeit
import numpy as np
import collections

import onnx
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
from tvm import autotvm

ANSOR_TYPE = "Ansor"
AUTO_TVM_TYPE = "AutoTVM"


@tvm.register_func("tvm_run_with_benchmark")
def run_with_benchmark(mod):
    run = mod.get_function('run')

    def benchmark(name):
        t = timeit.Timer(lambda: run()).repeat(repeat=5, number=5)
        ts = np.array(t) * 1000
        print("{} benchmark results: {:.2f}ms mean, {:.2f}ms median, {:.2f}ms std".format(
            name, np.mean(ts), np.median(ts), np.std(ts)
        ))
    if os.getenv("AUTOTVM_TUNING_LOG"):
        benchmark("Tuned")
    else:
        benchmark("Baseline")


@tvm.register_func("tvm_run")
def run_without_benchmark(mod):
    run = mod.get_function('run')
    run()


@tvm.register_func("tvm_onnx_import_and_compile")
def onnx_compile(model_string,
                 model_path,
                 target,
                 target_host,
                 opt_level,
                 opset,
                 freeze_params,
                 input_shapes,
                 nhwc=False,
                 tuning_logfile="",
                 tuning_type=AUTO_TVM_TYPE):
    model = onnx.load_model_from_string(bytes(model_string))
    if model_path:
        base_dir = os.path.dirname(os.path.abspath(model_path))
        onnx.load_external_data_for_model(model, base_dir)

    # Collect only feed input names from all input names
    all_input_names = [node.name for node in model.graph.input]
    all_initializer = [node.name for node in model.graph.initializer]
    net_feed_input_names = list(set(all_input_names) - set(all_initializer))

    # Match names and input shapes
    all_input_mapping = [(name, shape) for (name, shape) in zip(all_input_names, input_shapes)]
    # Using an ordereddict maintains input ordering.
    shape_dict = collections.OrderedDict(all_input_mapping)
    # Get only feed input pairs
    feed_shape_dict = {}
    for name in net_feed_input_names:
        feed_shape_dict[name] = shape_dict[name]

    irmod, params = relay.frontend.from_onnx(model, feed_shape_dict, opset=opset, freeze_params=freeze_params)

    # TODO(vvchernov): replace prints by logger, but investigate ORT logging system for python before
    # Also see lines 91, 106
    # print("Build TVM graph executor")
    # Tuning file can be set by client through ep options
    if tuning_logfile == "":
        tuning_logfile = os.getenv("AUTOTVM_TUNING_LOG")
    if tuning_type == ANSOR_TYPE:
        if tuning_logfile:
            desired_layouts = {
                "nn.conv2d": ["NHWC", "default"],
                "nn.conv2d_transpose": ["NHWC", "default"],
                "nn.upsampling": ["NHWC", "default"],
                "vision.roi_align": ["NHWC", "default"],
            }
            # print("Use tuning file from ", ANSOR_TYPE, ": ", tuning_logfile)
            with auto_scheduler.ApplyHistoryBest(tuning_logfile):
                with tvm.transform.PassContext(opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}):
                    if nhwc:
                        irmod = relay.transform.InferType()(irmod)
                        model_nhwc = relay.transform.ConvertLayout(desired_layouts)(irmod)
                        model_nhwc = tvm.relay.transform.EliminateCommonSubexpr()(model_nhwc)
                        irmod = tvm.relay.transform.FoldConstant()(model_nhwc)
                    lib = relay.build(irmod, target=target, target_host=target_host)
        else:
            with tvm.transform.PassContext(opt_level=opt_level):
                lib = relay.build(irmod, target=target, target_host=target_host, params=params)
    elif tuning_type == AUTO_TVM_TYPE:
        with relay.build_config(opt_level=opt_level):
            if tuning_logfile:
                # print("Use tuning file from ", AUTO_TVM_TYPE, ": ", tuning_logfile)
                with autotvm.apply_history_best(tuning_logfile):
                    # XXX: do not pass parameters to relay.build otherwise they will be inline into the module
                    lib = relay.build(irmod, target_host=target_host, target=target)
            else:
                lib = relay.build(irmod, target_host=target_host, target=target)
    else:
        # TODO(vvchernov): replace prints by logger, but investigate ORT logging system for python before
        # print is not commented out while it declares error
        print("ERROR: Tuning log type {} is unsupported. ".format(tuning_type),
              "Only {} and {} types are supported".format(ANSOR_TYPE, AUTO_TVM_TYPE))
        return None

    ctx = tvm.device(target, 0)
    m = graph_executor.GraphModule(lib["default"](ctx))
    return m.module
