import math
import os
import sys
import subprocess
import copy
import numpy as np
import torch
import onnx

from onnxruntime.training import optim

def is_windows():
    return sys.platform.startswith("win")

def run_subprocess(args, cwd=None, capture=False, dll_path=None,
                   shell=False, env={}, log=None):
    if log:
        log.info("Running subprocess in '{0}'\n{1}".format(
            cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (
        None, None)
    my_env.update(env)
    completed_process = subprocess.run(
        args, cwd=cwd, check=True, stdout=stdout, stderr=stderr,
        env=my_env, shell=shell)
    
    if log:
        log.debug("Subprocess completed. Return code=" +
                str(completed_process.returncode))
    return completed_process


def legacy_constant_lr_scheduler(global_step, initial_lr, total_steps, warmup):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        new_lr = initial_lr
    return new_lr


def legacy_cosine_lr_scheduler(global_step, initial_lr, total_steps, warmup, cycles):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        progress = float(global_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        new_lr = initial_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * progress)))
    return new_lr



def legacy_linear_lr_scheduler(global_step, initial_lr, total_steps, warmup):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    else:
        new_lr = initial_lr * max(0.0, float(total_steps - global_step) / float(max(1, total_steps - num_warmup_steps)))
    return new_lr


def legacy_poly_lr_scheduler(global_step, initial_lr, total_steps, warmup, power, lr_end):
    num_warmup_steps = warmup * total_steps
    if global_step < num_warmup_steps:
        new_lr = initial_lr * float(global_step) / float(max(1, num_warmup_steps))
    elif global_step > total_steps:
        new_lr = lr_end
    else:
        lr_range = initial_lr - lr_end
        decay_steps = total_steps - num_warmup_steps
        pct_remaining = 1 - (global_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        new_lr = decay
    return new_lr


def generate_dummy_optim_state(model, optimizer):
    np.random.seed(0)
    if not (isinstance(optimizer, optim.AdamConfig) or isinstance(optimizer, optim.LambConfig)):
        return dict()

    moment_keys = ["Moment_1", "Moment_2"]
    uc_key = "Update_Count"
    step_key = "Step"
    shared_state_key = "shared_optimizer_state"

    optim_state = dict()
    weight_shape_map = dict()
    if isinstance(model, torch.nn.Module):
        weight_shape_map = {name: param.size() for name, param in model.named_parameters()}
    elif isinstance(model, onnx.ModelProto):
        weight_shape_map = {n.name: n.dims for n in model.graph.initializer}
    else:
        raise ValueError("'model' must be either 'torch.nn.Module' or 'onnx.ModelProto'")

    for weight_name, weight_shape in weight_shape_map.items():
        per_weight_state = dict()
        for moment in moment_keys:
            per_weight_state[moment] = np.random.uniform(-2, 2, weight_shape).astype(np.float32)
        if isinstance(optimizer, optim.AdamConfig):
            per_weight_state[uc_key] = np.full([1], 5, dtype=np.int64)
        optim_state[weight_name] = copy.deepcopy(per_weight_state)
    if isinstance(optimizer, optim.LambConfig):
        step_val = np.full([1], 5, dtype=np.int64)
        optim_state[shared_state_key] = {step_key: step_val}
    return optim_state


def get_optim_state_from_state_dict(state_dict, optimizer):
    if not (isinstance(optimizer, optim.AdamConfig) or isinstance(optimizer, optim.LambConfig)):
        return dict()

    moment_keys = ["Moment_1", "Moment_2"]
    uc_key = "Update_Count"
    step_key = "Step"
    shared_state_key = "shared_optimizer_state"

    optim_state = dict()
    for param_name, v in state_dict.items():
        if '_view_' in param_name:
            param_name = param_name.split('_view_')[0]
            print("param name: ", param_name)

        for moment in moment_keys:
            if param_name.startswith(moment):
                fp32_name = param_name.split(moment + '_')[-1]
                if fp32_name not in optim_state:
                    optim_state[fp32_name] = dict()
                optim_state[fp32_name].update({moment: v})
                break
        if param_name.startswith(uc_key):
            fp32_name = param_name.split(uc_key + '_')[-1]
            if fp32_name not in optim_state:
                optim_state[fp32_name] = dict()
            optim_state[fp32_name].update({uc_key: v})
        elif param_name == step_key:
            optim_state[shared_state_key] = {step_key: v}
    return optim_state
