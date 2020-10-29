import math


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
