import onnx
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription
from onnxruntime.capi.ort_trainer import LossScaler
import torch

def setup_onnxruntime_with_mpi(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    args.local_rank = comm.Get_rank() % torch.cuda.device_count()
    args.world_rank = comm.Get_rank()
    args.world_size=comm.Get_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.n_gpu = 1

    from onnxruntime.capi._pybind_state import set_cuda_device_id 
    set_cuda_device_id(args.local_rank)

    from onnxruntime.capi._pybind_state import set_arena_extend_strategy, ArenaExtendStrategy
    set_arena_extend_strategy(ArenaExtendStrategy.kSameAsRequested)

    return device

def bert_model_description(args):
    vocab_size = 30528
    # input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch',], torch.int64, num_classes = 2)
    micro_batch = args.train_batch_size // args.gradient_accumulation_steps
    input_ids_desc = IODescription('input_ids', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = vocab_size)
    segment_ids_desc = IODescription('segment_ids', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = 2)
    input_mask_desc = IODescription('input_mask', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = 2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = vocab_size)
    next_sentence_labels_desc = IODescription('next_sentence_labels', [args.train_batch_size,2], torch.int64, num_classes = 2)
    loss_desc = IODescription('loss', [], torch.float32)
    return ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc, next_sentence_labels_desc], [loss_desc])

# opset 12
# from ort_supplement.onnx_transforms.model_transform import add_name, fix_transpose, add_expand_shape, process_concat, process_dropout #, handle_expand_input_is_not_constant_case, fix_dim, fix_expand

# opset 10
from ort_supplement.onnx_transforms.model_transform import add_name, fix_transpose, add_expand_shape, process_concat, process_dropout, handle_expand_input_is_not_constant_case, fix_dim, fix_expand

from ort_supplement.onnx_transforms.layer_norm_transform import layer_norm_transform

def postprocess_model(model):
    add_name(model)

    # not in TJ branch process_concat(model)

    # opset 10:
    handle_expand_input_is_not_constant_case(model)
    fix_expand(model)
    fix_dim(model)
    process_dropout(model)
    # --- 

    # not in TJ branch fix_transpose(model)
    add_expand_shape(model)
    layer_norm_transform(model)

def create_ort_trainer(args, device, model):
    # set GPU memory limitation
    from onnxruntime.capi._pybind_state import set_cuda_mem_limit
    ort_cuda_mem_limit_in_gbs = 16
    set_cuda_mem_limit(int(ort_cuda_mem_limit_in_gbs * 1024 * 1024 *1024))

    # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
    def map_optimizer_attributes(name):
        no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
        no_decay = False
        for no_decay_key in no_decay_keys:
            if no_decay_key in name:
                no_decay = True
                break
        if no_decay:
            return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
        else:
            return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

    # we request ORTTrainer to create a LambOptimizer with given optimizer_attributes. 
    # train_step does forward, backward, and optimize step.
    model = ORTTrainer(model, None, bert_model_description(args), "LambOptimizer", 
        map_optimizer_attributes,
        IODescription('Learning_Rate', [1,], torch.float32),
        device, postprocess_model=postprocess_model, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_rank=args.world_rank, world_size=args.world_size,
        use_mixed_precision = True if args.fp16 else False,
        allreduce_post_accumulation = True if args.allreduce_post_accumulation else False,
        partition_optimizer=True,
        _opset_version = 10)

    return model

from ort_supplement.lr_schedules import SCHEDULES
def get_lr(args, training_steps, schedule='warmup_poly'):
    if args.max_steps == -1:
        return args.learning_rate

    schedule_fct = SCHEDULES[schedule]
    return args.learning_rate * schedule_fct(training_steps / args.max_steps, args.warmup_proportion)

def run_ort_training_step(args, global_step, training_steps, model, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    if args.fp16:
        loss_scaler = LossScaler(model.loss_scale_input_name, True, up_scale_window=2000)

    lr = get_lr(args, global_step, args.schedule)
    learning_rate = torch.tensor([lr])
    if args.fp16:
        loss_scale = torch.tensor([loss_scaler.loss_scale_])
        loss = model.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate, loss_scale)
        all_finite = 1
        if isinstance(loss, (list, tuple)):
            assert len(loss) == 2
            loss, all_finite = loss
    else:
        loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate)
    if training_steps % args.gradient_accumulation_steps == 0:
        if args.fp16:
            loss_scaler.update_loss_scale(all_finite.item())
        global_step += 1

    return loss, global_step