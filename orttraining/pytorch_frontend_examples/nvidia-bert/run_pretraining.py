# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp
import multiprocessing

from tokenization import BertTokenizer
import modeling
from apex.optimizers import FusedLAMB
from schedulers import PolyWarmUpScheduler

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process, format_step
from apex.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler
from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state

import dllogger
from concurrent.futures import ProcessPoolExecutor

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0

# --- ort training edit
import onnx
# ---

#Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    # --- ort training edit: we need to skip last batch when hard coding inputs as an optimization
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, 
                                  num_workers=4, worker_init_fn=worker_init,
                                  pin_memory=True, drop_last=args.use_ort_trainer)
    # ---
    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]
class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size, batch_size, seq_length):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view([self.batch_size * self.seq_length, self.vocab_size]), masked_lm_labels.view(self.batch_size * self.seq_length))
        next_sentence_loss = self.loss_fn(seq_relationship_score, next_sentence_labels.view(self.batch_size))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

# --- ort training edit
# we manually add the loss function into the bert model
# currently ort front end support for this assumes a single tensor input for labels
class bert_model_with_loss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(bert_model_with_loss, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn

    def forward(self, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels):
        preds_score, seq_relation_score = self.model_(input_ids, segment_ids, input_mask)
        return self.loss_fn_(preds_score, seq_relation_score, masked_lm_labels, next_sentence_labels)
# ---

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    # --- ort training edit
    parser.add_argument('--use_ort_trainer',
                        default=False,
                        action='store_true',
                        help="Whether to run with ort in fully optimized mode (run optimization in ort as opposed in pytorch).")
    parser.add_argument('--schedule',
                        default='warmup_poly',
                        type=str)
    # ---
    args = parser.parse_args()
    
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    # --- ort training edit
    if args.use_ort_trainer:
        global ort_supplement
        import ort_supplement.ort_supplement as ort_supplement
        device = ort_supplement.setup_onnxruntime_with_mpi(args)
    # ---
    elif args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = 1
        
    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # --- ort training edit
    if not args.use_ort_trainer:
        modeling.ACT2FN["bias_gelu"] = torch.jit.script(modeling.ACT2FN["bias_gelu"])
    # ---

    model = modeling.BertForPreTraining(config)
    criterion = BertPretrainingCriterion(config.vocab_size, args.train_batch_size, args.max_seq_length)

    # --- ort training edit
    if args.use_ort_trainer:
        model.enable_apex(False)
        model = bert_model_with_loss(model, criterion)
        model = ort_supplement.create_ort_trainer(args, device, model)
    # ---

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")

        model.load_state_dict(checkpoint['model'], strict=False)
        
        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step)

    # --- ort training edit (note: returns early)
    if args.use_ort_trainer:
        return model, None, None, checkpoint, global_step, None
    # ---

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters, 
                          lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                       warmup=args.warmup_proportion, 
                                       total_steps=args.max_steps)
    if args.fp16:

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", cast_model_outputs=torch.float16)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale, cast_model_outputs=torch.float16)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    model.checkpoint_activations(args.checkpoint_activations)

    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

        # Restore AMP master parameters          
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    global skipped_steps
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        loss_scale = _amp_state.loss_scalers[0].loss_scale() if args.fp16 else 1
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (torch.distributed.get_world_size() * args.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./loss_scale)
        # 5. update loss scale
        if args.fp16:
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
        else:
            had_overflow = 0
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            skipped_steps += 1
            if is_main_process():
                scaler = _amp_state.loss_scalers[0]
                dllogger.log(step="PARAMETER", data={"loss_scale": scaler.loss_scale()})
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step

def main():

    # --- sukha edit
    already_saved_onnx_model = False
    # ---
    args = parse_arguments()

    if args.use_env and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        
    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)

    device, args = setup_training(args)
    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, criterion = prepare_model_and_optimizer(args, device)

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    raw_train_start = time.time()
    if args.do_train:
        if is_main_process():
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
            dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0

        pool = ProcessPoolExecutor(1)

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)


            shared_file_list = {}

            # --- ort training edit

            # if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
            #     remainder = torch.distributed.get_world_size() % num_files
            #     data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_start_id)%num_files]
            # else:
            #     data_file = files[(f_start_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                world_rank = torch.distributed.get_rank()
            elif hasattr(args, 'world_size'):
                world_size = args.world_size
                world_rank = args.world_rank
            else:
                world_size = 1
                world_rank = 0

            if world_size > num_files:
                remainder = world_size % num_files
                data_file = files[(f_start_id*world_size + world_rank + remainder*f_start_id)%num_files]
            elif world_size > 1:
                data_file = files[(f_start_id*world_size + world_rank)%num_files]
            else:
                data_file = files[f_start_id % num_files]
            # ---

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            # --- ort training edit: we need to skip last batch when we hard code inputs as an optimization
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size * args.n_gpu,
                                          num_workers=4, worker_init_fn=worker_init,
                                          pin_memory=True, drop_last=args.use_ort_trainer)
            # ---
            # shared_file_list["0"] = (train_dataloader, data_file)

            overflow_buf = None
            # --- ort training edit
            gpu_batch_size = args.train_batch_size // args.gradient_accumulation_steps
            if args.allreduce_post_accumulation and not args.use_ort_trainer:
                overflow_buf = torch.cuda.IntTensor([0])
            # ---
            
            if len(files) == 1:
                f_start_id = -1
            for f_id in range(f_start_id + 1 , len(files)):
                
                # --- ort training edit
                # if torch.distributed.get_world_size() > num_files:
                #     data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank() + remainder*f_id)%num_files]
                # else:
                #     data_file = files[(f_id*torch.distributed.get_world_size()+torch.distributed.get_rank())%num_files]

                if world_size > num_files:
                    data_file = files[(f_id*world_size+world_rank + remainder*f_id)%num_files]
                elif world_size > 1:
                    data_file = files[(f_id*world_size + world_rank)%num_files]
                else:
                    data_file = files[f_id%num_files]
                # ---

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init)

                train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process() else train_dataloader
                for step, batch in enumerate(train_iter):

                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    divisor = args.gradient_accumulation_steps

                    # --- ort training edit
                    if args.use_ort_trainer:
                        loss, global_step = ort_supplement.run_ort_training_step(args, global_step, training_steps, model, batch)
                    # ----
                    else:
                        prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                        loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.

                        if args.gradient_accumulation_steps > 1:
                            if not args.allreduce_post_accumulation:
                                # this division was merged into predivision
                                loss = loss / args.gradient_accumulation_steps
                                divisor = 1.0
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    average_loss += loss.item()

                    # --- ort training edit
                    if not args.use_ort_trainer:
                    # ---
                        if training_steps % args.gradient_accumulation_steps == 0:
                            lr_scheduler.step()  # learning rate warmup
                            global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)

                    if global_step >= args.max_steps:
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if is_main_process():
                            dllogger.log(step=(epoch, global_step, ), data={"final_loss": final_loss})
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if is_main_process():
                            data = {"average_loss": average_loss / (args.log_freq * divisor),
                                    "step_loss": loss.item() * args.gradient_accumulation_steps / divisor}
                            # --- ort training edit
                            if not args.use_ort_trainer:
                                data["learning_rate"] = optimizer.param_groups[0]['lr']
                            # ---
                            dllogger.log(step=(epoch, global_step, ), data=data)
                        average_loss = 0

                    if global_step >= args.max_steps or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                        if is_main_process() and not args.skip_checkpoint:
                            # Save a trained model
                            dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                            else:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                            if args.do_train:
                                state = {'model': model_to_save.state_dict(),
                                         'files': [f_id] + files}
                                # --- ort training edit
                                if not args.use_ort_trainer:
                                    state['optimizer'] = optimizer.state_dict()
                                    state['master params'] = list(amp.master_params(optimizer))
                                # ---
                                torch.save(state, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps:
                            # --- ort training edit
                            if is_main_process() and args.use_ort_trainer:
                                print('-----------------------save onnx model-----------------------')
                                if not args.phase2:
                                    model_to_save.save_as_onnx('{}/phase1_bert.onnx'.format(args.output_dir))
                                else:
                                    model_to_save.save_as_onnx('{}/final_bert.onnx'.format(args.output_dir))
                            # ---
                            del train_dataloader
                            # thread.join()
                            return args, final_loss, train_time_raw

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1


if __name__ == "__main__":

    now = time.time()
    args, final_loss, train_time_raw = main()
    gpu_count = args.n_gpu
    args.max_steps += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
    if args.resume_step == -1:
        args.resume_step = 0
    if torch.distributed.is_initialized():
        gpu_count = torch.distributed.get_world_size()
    if is_main_process():
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * args.gradient_accumulation_steps * gpu_count\
                        * (args.max_steps - args.resume_step + skipped_steps) / train_time_raw
        dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time, "training_sequences_per_second": training_perf,
                                         "final_loss": final_loss, "raw_train_time": train_time_raw })
    dllogger.flush()
