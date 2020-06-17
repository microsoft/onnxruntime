import torch

from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription

from orttraining_test_data_loader import create_ort_test_dataloader, BatchArgsOption, split_batch
from orttraining_test_bert_postprocess import postprocess_model

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)

def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}

def get_lr(args, training_steps, schedule='warmup_poly'):
    if args.max_steps == -1:
        return args.learning_rate

    schedule_fct = SCHEDULES[schedule]
    return args.learning_rate * schedule_fct(training_steps / args.max_steps, args.warmup_proportion)

def map_optimizer_attributes(name):
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

def run_test(model, model_desc, device, args, gradient_accumulation_steps, fp16,
    allreduce_post_accumulation, get_lr_this_step, use_internal_get_lr_this_step, loss_scaler, use_internal_loss_scaler,
    batch_args_option):
    dataloader = create_ort_test_dataloader(model_desc.inputs_, args.batch_size, args.seq_len, device)

    model = ORTTrainer(model, None, model_desc, "LambOptimizer",
        map_optimizer_attributes=map_optimizer_attributes,
        learning_rate_description=IODescription('Learning_Rate', [1,], torch.float32),
        device=device,
        _enable_internal_postprocess=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
        world_rank=args.local_rank, world_size=args.world_size,
        use_mixed_precision=fp16,
        allreduce_post_accumulation=allreduce_post_accumulation,
        get_lr_this_step=get_lr_this_step if use_internal_get_lr_this_step else None,
        loss_scaler=loss_scaler if use_internal_loss_scaler else None,
        _opset_version=12)

    # trainig loop
    eval_batch = None
    model.train()
    for step, batch in enumerate(dataloader):
        if eval_batch is None:
            eval_batch = batch

        if not use_internal_get_lr_this_step:
            lr = get_lr_this_step(step)
            learning_rate = torch.tensor([lr])

        if not use_internal_loss_scaler and fp16:
            loss_scale = torch.tensor([loss_scaler.loss_scale_])

        if batch_args_option == BatchArgsOption.List:
            if not use_internal_get_lr_this_step:
                batch = batch + [learning_rate, ]
            if not use_internal_loss_scaler and fp16:
                batch = batch + [loss_scale, ]
            outputs = model(*batch)
        elif batch_args_option == BatchArgsOption.Dict:
            args, kwargs = split_batch(batch, model_desc.inputs_, 0)
            if not use_internal_get_lr_this_step:
                kwargs['Learning_Rate'] = learning_rate
            if not use_internal_loss_scaler and fp16:
                kwargs[model.loss_scale_input_name] = loss_scale
            outputs = model(*args, **kwargs)
        else:
            args_count = int(len(model_desc.inputs_) / 2)   # approx helf args, half kwargs
            args, kwargs = split_batch(batch, model_desc.inputs_, args_count)
            if not use_internal_get_lr_this_step:
                kwargs['Learning_Rate'] = learning_rate
            if not use_internal_loss_scaler and fp16:
                kwargs[model.loss_scale_input_name] = loss_scale
            outputs = model(*args, **kwargs)

        print(outputs[0])

    # eval
    model.eval()
    if batch_args_option == BatchArgsOption.List:
        outputs = model(*batch)
    elif batch_args_option == BatchArgsOption.Dict:
        args, kwargs = split_batch(batch, model_desc.inputs_, 0)
        outputs = model(*args, **kwargs)
    else:
        args_count = int(len(model_desc.inputs_) / 2)   # approx helf args, half kwargs
        args, kwargs = split_batch(batch, model_desc.inputs_, args_count)
        outputs = model(*args, **kwargs)

    return (output.cpu().numpy() for output in outputs)

