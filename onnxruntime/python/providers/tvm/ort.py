# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import collections
import copy
import logging
import os

import onnx
import tvm
from tvm import auto_scheduler, autotvm, relay
from tvm.contrib import graph_executor
from tvm.relay import vm

log = logging.getLogger("tvm_ep")

ANSOR_TYPE = "Ansor"
AUTO_TVM_TYPE = "AutoTVM"


@tvm.register_func("tvm_onnx_import_and_compile")
def onnx_compile(
    model_string,
    model_path,
    executor,
    target,
    target_host,
    opt_level,
    opset,
    freeze_params,
    input_shapes,
    nhwc=False,
    tuning_logfile="",
    tuning_type=AUTO_TVM_TYPE,
):
    def get_tvm_executor(irmod, executor, target, params):
        if executor == "vm":
            log.info("Build TVM virtual machine")
            lib = vm.compile(
                copy.deepcopy(irmod),
                target,
                params=params,
            )
        elif executor == "graph":
            log.info("Build TVM graph executor")
            lib = relay.build(irmod, target=target, params=params)
        else:
            log.error(f'Executor type {executor} is unsupported. Only "vm" and "graph" types are supported')
            return None
        return lib

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
    irmod = relay.transform.DynamicToStatic()(irmod)

    # Tuning file can be set by client through ep options
    if not tuning_logfile:
        tuning_logfile = os.getenv("AUTOTVM_TUNING_LOG")
    lib = None
    tvm_target = tvm.target.Target(target, host=target_host)
    if tuning_logfile:
        if tuning_type == ANSOR_TYPE:
            desired_layouts = {
                "nn.conv2d": ["NHWC", "default"],
                "nn.conv2d_transpose": ["NHWC", "default"],
                "nn.upsampling": ["NHWC", "default"],
                "vision.roi_align": ["NHWC", "default"],
            }
            log.info("Use tuning file from %s: %s", ANSOR_TYPE, tuning_logfile)
            with auto_scheduler.ApplyHistoryBest(tuning_logfile):  # noqa: SIM117
                with tvm.transform.PassContext(
                    opt_level=opt_level,
                    config={
                        "relay.backend.use_auto_scheduler": True,
                        "relay.FuseOps.max_depth": 30,
                    },
                ):
                    if nhwc:
                        seq = tvm.transform.Sequential(
                            [
                                relay.transform.InferType(),
                                relay.transform.ConvertLayout(desired_layouts),
                                relay.transform.EliminateCommonSubexpr(),
                                relay.transform.FoldConstant(),
                            ]
                        )
                        irmod = seq(irmod)
                    lib = get_tvm_executor(irmod, executor, tvm_target, params)
        elif tuning_type == AUTO_TVM_TYPE:
            with relay.build_config(opt_level=opt_level):
                log.info("Use tuning file from %s: %s", AUTO_TVM_TYPE, tuning_logfile)
                with autotvm.apply_history_best(tuning_logfile):
                    lib = get_tvm_executor(irmod, executor, tvm_target, params)
        else:
            log.error(
                f"Tuning log type {tuning_type} is unsupported. "
                f"Only {ANSOR_TYPE} and {AUTO_TVM_TYPE} types are supported"
            )
            return None
    else:
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = get_tvm_executor(irmod, executor, tvm_target, params)

    if lib is None:
        return None

    ctx = tvm.device(target, 0)
    if executor == "vm":
        m = tvm.runtime.vm.VirtualMachine(lib, ctx)
    elif executor == "graph":
        m = graph_executor.GraphModule(lib["default"](ctx))
    else:
        print(
            f"ERROR: Executor type {executor} is unsupported. ",
            'Only "vm" and "graph" types are supported',
        )
        return None

    return m.module
