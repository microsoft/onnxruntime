import math
import os
import sys
import subprocess

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
