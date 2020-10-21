from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import os
import logging
import random
# import h5py
from tqdm import tqdm
import datetime
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import json

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary, iterators, data_utils
from fairseq.logging import meters, metrics, progress_bar

from concurrent.futures import ProcessPoolExecutor

import onnxruntime as ort
from onnxruntime.training import amp, optim, orttrainer
from onnxruntime.training.optim import PolyWarmupLRScheduler, LinearWarmupLRScheduler

# need to override torch.onnx.symbolic_opset12.nll_loss to handle ignore_index == -100 cases.
# the fix for ignore_index == -100 cases is already in pytorch master.
# however to use current torch master is causing computation changes in many tests.
# eventually we will use pytorch with fixed nll_loss once computation
# issues are understood and solved.
import onnxruntime.capi.pt_patch

# we cannot make full convergence run in nightly pipeling because of its timeout limit,
# max_steps is still needed to calculate learning rate. force_to_stop_max_steps is used to
# terminate the training before the pipeline run hit its timeout.
force_to_stop_max_steps = 2500

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process(args):
    if hasattr(args, 'world_rank'):
        return args.world_rank in [-1, 0]
    else:
        return get_rank() == 0

def bart_model_description(args):
    batch = args.train_batch_size
    max_seq_len_in_batch = args.seq_len
    
    new_model_desc = {
        'inputs': [
            ('src_tokens', [batch, max_seq_len_in_batch],),
            ('prev_output_tokens', [batch, max_seq_len_in_batch],),
            ('target', [batch*max_seq_len_in_batch],)],
        'outputs': [
            ('loss', [], True)]}
    return new_model_desc

def get_train_iterator(
    task,
    args,
    epoch,
    combine=True,
    load_dataset=True,
    data_selector=None,
    shard_batch_itr=True,
    disable_iterator_cache=False,
):
    """Return an EpochBatchIterator over the training set for a given epoch."""
    if load_dataset:
        logger.info("loading train data for epoch {}".format(epoch))
        task.load_dataset(
            args.train_subset,
            epoch=epoch,
            combine=combine,
            data_selector=data_selector,
        )
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=args.seq_len,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.world_size if shard_batch_itr else 1,
        shard_id=args.world_rank if shard_batch_itr else 0,
        num_workers=args.num_workers,
        epoch=epoch,
        data_buffer_size=args.data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
    )
    return batch_iterator

class BARTModelWithLoss(torch.nn.Module):
    def __init__(self, model, padding_idx):
        super(BARTModelWithLoss, self).__init__()
        self.model_ = model
        self.padding_idx_ = padding_idx

    def forward(self, src_tokens, prev_output_tokens, target):
        src_lengths = None
        net_output = self.model_(src_tokens, src_lengths, prev_output_tokens, features_only=False, classification_head_name=None)
        net_out = net_output[0]

        # flatten the net_out, merging the first two dims
        net_out = net_out.view(-1, net_out.size(-1))

        lprobs = F.log_softmax(net_out.float(), dim=-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx_,
            reduction='sum',
        )
        return loss

def add_ort_args(parser):
    group = parser.add_argument_group("ORT")

    group.add_argument("--input_dir", default=None, metavar="DIR",
                       help="The input directory where the model data is present.")
    group.add_argument("--output_dir", default=None, metavar="DIR",
                       help="The output directory where the model checkpoints will be written.")
    group.add_argument("--train_batch_size", metavar="N", default=32, type=int,
                       help="Batch size for training")
    group.add_argument("--gradient_accumulation_steps", metavar="N", default=1, type=int,
                       help="Number of updates steps to accumualte before performing a backward/update pass.")
    group.add_argument("--allreduce_post_accumulation", action="store_true",
                       help="Whether to do fp16 allreduce post accumulation.")
    group.add_argument("--local_rank", metavar="N", default=-1, type=int,
                       help="local_rank for distributed training on gpus.")
    group.add_argument("--world_rank", metavar="N", default=-1, type=int,
                       help="world_rank for distributed training on gpus.")
    group.add_argument("--world_size", metavar="N", default=1, type=int,
                       help="world_size for distributed training on gpus.")
    group.add_argument("--max_steps", metavar="N", default=1000, type=int,
                       help="Total number of training steps to perform.")
    group.add_argument("--force_num_layers", metavar="N", default=None, type=int,
                       help="Reduced number of layers.")
    group.add_argument("--padding_idx", metavar="N", default=-100, type=int,
                       help="Index of padding token.")
    group.add_argument("--seq_len", metavar="N", default=1024, type=int,
                       help="Source and target seq lengths.")
    group.add_argument("--log_freq", metavar="N", default=1, type=int,
                       help="Logging frequency.")
    group.add_argument('--warmup_proportion', default=0.01, type=float,
                       help='Proportion of training to perform linear learning rate warmup for. \
            E.g., 0.1 = 10%% of training.')
    return parser

def to_json_string(args):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(vars(args), indent=2)

def to_sanitized_dict(args) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = vars(args)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

def to_list(args) -> list:
        """
        List to use with Argparser.parse_known_args
        """
        d = args
        o=[]
        for k,v in d.items():
            o.extend(["--"+ k , str(v)])
        return o


def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        args.local_rank = 0
        args.world_rank = 0

    print("args.local_rank: ", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.n_gpu = 1

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    # args.train_batch_size is per global step (optimization step) batch size
    # now make it a per gpu batch size
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.train_batch_size = args.train_batch_size // args.world_size
    args.max_sentences = args.train_batch_size

    logger.info("setup_training: args.train_batch_size = %d", args.train_batch_size)
    return device, args


def prepare_model(args, device):
    if args.force_num_layers is not None:
        logger.info("forcing num layers to %d", args.force_num_layers)
        args.encoder_layers = args.force_num_layers
        args.decoder_layers = args.force_num_layers

    task = tasks.setup_task(args)

    tgt_dict = Dictionary.load(
            os.path.join(args.input_dir, "dict.{}.txt".format(args.target_lang))
        )
    args.padding_idx = tgt_dict.pad()
    model = task.build_model(args)

    model = BARTModelWithLoss(model, args.padding_idx)
    model_desc = bart_model_description(args)

    lr_scheduler = LinearWarmupLRScheduler(total_steps=int(args.max_steps), warmup=args.warmup_proportion)

    loss_scaler = amp.DynamicLossScaler() if args.fp16 else None

    options = orttrainer.ORTTrainerOptions({'batch': {
                                                'gradient_accumulation_steps': args.gradient_accumulation_steps},
                                            'device': {'id': str(device)},
                                            'mixed_precision': {
                                                'enabled': args.fp16,
                                                'loss_scaler': loss_scaler},
                                            'debug': {'deterministic_compute': True, },
                                            'utils': {
                                                'grad_norm_clip': True},
                                            'distributed': {
                                                'world_rank': max(0, args.local_rank),
                                                'world_size': args.world_size,
                                                'local_rank': max(0, args.local_rank),
                                                'horizontal_parallel_size' : args.world_size,
                                                'allreduce_post_accumulation': args.allreduce_post_accumulation},
                                            'lr_scheduler': lr_scheduler
                                            })

    param_optimizer = list(model.named_parameters())
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    params = [{
        'params': [n for n, p in param_optimizer if any(no_decay_key in n for no_decay_key in no_decay_keys)],
        "alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}, {
        'params': [n for n, p in param_optimizer if not any(no_decay_key in n for no_decay_key in no_decay_keys)],
        "alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}]

    optim_config = optim.AdamConfig(params=params, lr=args.lr[0], do_bias_correction=True)
    model = orttrainer.ORTTrainer(model, model_desc, optim_config, options=options)

    return model, task

def pad_to_len(tokens, args):
    return data_utils.collate_tokens(tokens, args.padding_idx, pad_to_length=args.seq_len)

def do_pretrain(args):
    if is_main_process(args) and args.tensorboard_logdir:
        tb_writer = SummaryWriter(log_dir=args.tensorboard_logdir)
        tb_writer.add_text("args", to_json_string(args))
        tb_writer.add_hparams(to_sanitized_dict(args), metric_dict={})
    else:
        tb_writer = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ort.set_seed(args.seed)

    device, args = setup_training(args)

    model, task = prepare_model(args, device)

    logger.info("Running training: Batch size = %d, initial LR = %f", args.train_batch_size, args.lr[0])

    most_recent_ckpts_paths = []
    average_loss = 0.0
    epoch = 0
    training_steps = 0

    pool = ProcessPoolExecutor(1)
    while True:
        epoch_itr = get_train_iterator(
            task,
            args,
            1,
            # sharded data: get train iterator for next epoch
            load_dataset=True,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
            )
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
        )
        update_freq = (
            args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(args.update_freq)
            else args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        for step, batch in enumerate(progress):
            training_steps += 1

            net_input = batch[0]['net_input']
            src_tokens = pad_to_len(net_input['src_tokens'], args).to(device)
            prev_output_tokens = pad_to_len(net_input['prev_output_tokens'], args).to(device)
            target = pad_to_len(batch[0]['target'], args).view(-1).to(device)


            loss = model.train_step(src_tokens, prev_output_tokens, target)
            average_loss += loss.item()

            global_step = model._train_step_info.optimization_step
            if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                if is_main_process(args):
                    divisor = args.log_freq * args.gradient_accumulation_steps
                    if tb_writer:
                        lr = model.options.lr_scheduler.get_last_lr()[0]
                        tb_writer.add_scalar('train/summary/scalar/Learning_Rate', lr, global_step)
                        if args.fp16:
                            tb_writer.add_scalar('train/summary/scalar/loss_scale_25', loss, global_step)
                            # TODO: ORTTrainer to expose all_finite
                            # tb_writer.add_scalar('train/summary/scalar/all_fp16_gradients_finite_859', all_finite, global_step)
                        tb_writer.add_scalar('train/summary/total_loss', average_loss / divisor, global_step)
                    
                    print("Step:{} Average Loss = {}".format(global_step, average_loss / divisor))

                if global_step >= args.max_steps or global_step >= force_to_stop_max_steps:
                    if tb_writer:
                        tb_writer.close()

                    final_loss = average_loss / (args.log_freq * args.gradient_accumulation_steps)
                    return final_loss

                average_loss = 0

        del train_dataloader

        epoch += 1


def generate_tensorboard_logdir(root_dir): 
    current_date_time = datetime.datetime.today()

    dt_string = current_date_time.strftime('BERT_pretrain_%y_%m_%d_%I_%M_%S')
    return os.path.join(root_dir, dt_string)


class ORTBertPretrainTest():
    def setUp(self):
        self.output_dir = '/tmp/bert_pretrain_results'
        self.bert_model = 'bert-base-uncased'
        self.max_steps = 300000
        self.lr = 2e-5
        self.data = '/bert_data/megatron_bart/bin_small/'
        self.fp16 = True
        self.allreduce_post_accumulation = True
        self.tensorboard_dir = '/tmp/bert_pretrain_results'

    def test_pretrain_throughput(self):
        self.train_batch_size = 2
        self.gradient_accumulation_steps = 1

        logger.info("self.gradient_accumulation_steps = %d", self.gradient_accumulation_steps)

        # only to run on few optimization step because we only want to make sure there is no throughput regression
        self.max_steps = 10
        args = {
            "arch" : 'bart_base',
            "task" : 'translation',
            "input_dir" : self.data,
            "local_rank" : self.local_rank,
            "world_rank" : self.world_rank,
            "world_size" : self.world_size,
            "max_steps" : self.max_steps,
            "lr" : self.lr,
            "train_batch_size" : self.train_batch_size,
            "gradient_accumulation_steps" : self.gradient_accumulation_steps,
            "fp16" : self.fp16,
            "allreduce_post_accumulation" : self.allreduce_post_accumulation,
            "tensorboard-logdir" : self.tensorboard_dir,
            "output_dir" : self.output_dir,
            "force_num_layers" : 1,
            }
        args = to_list(args)
        parser = options.get_training_parser()
        add_ort_args(parser)
        args, extras = options.parse_args_and_arch(parser, [self.data] + args, parse_known=True)
        do_pretrain(args)


# to do parallel training:
# python -m torch.distributed.launch --nproc_per_node 4 orttraining_run_bert_pretrain.py
if __name__ == "__main__":
    import sys
    logger.warning("sys.argv: %s", sys.argv)
    test = ORTBertPretrainTest()
    test.setUp()
    
    # usage:
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py
    # pytorch.distributed.launch will not work because ort backend requires MPI to broadcast ncclUniqueId
    #
    # calling unpublished get_mpi_context_xxx to get rank/size numbers.
    from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size
    world_size = get_mpi_context_world_size()
    if world_size > 1:
        print ('get_mpi_context_world_size(): ', world_size)
        local_rank = get_mpi_context_local_rank()

        if local_rank == 0:
            print('================================================================> os.getpid() = ', os.getpid())

        test.local_rank = local_rank
        test.world_rank = local_rank
        test.world_size = world_size
    
    test.test_pretrain_throughput()
