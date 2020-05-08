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


#==================
import csv
import os
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils import is_main_process, format_step
import math
import time

from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig

# from fused_adam_local import FusedAdamBert
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import dllogger


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        self.input_ids = np.asarray(f["input_ids"][:]).astype(np.int64)#[num_instances x max_seq_length])
        self.input_masks = np.asarray(f["input_mask"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.segment_ids = np.asarray(f["segment_ids"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.masked_lm_positions = np.asarray(f["masked_lm_positions"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.masked_lm_ids= np.asarray(f["masked_lm_ids"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.next_sentence_labels = np.asarray(f["next_sentence_labels"][:]).astype(np.int64) # [num_instances]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

    def __getitem__(self, index):
        
        input_ids= torch.from_numpy(self.input_ids[index]) # [max_seq_length]
        input_mask = torch.from_numpy(self.input_masks[index]) #[max_seq_length]
        segment_ids = torch.from_numpy(self.segment_ids[index])# [max_seq_length]
        masked_lm_positions = torch.from_numpy(self.masked_lm_positions[index]) #[max_pred_length]
        masked_lm_ids = torch.from_numpy(self.masked_lm_ids[index]) #[max_pred_length]
        next_sentence_labels = torch.from_numpy(np.asarray(self.next_sentence_labels[index])) #[1]
         
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        if len((masked_lm_positions == 0).nonzero()) != 0:
          index = (masked_lm_positions == 0).nonzero()[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]

def main():    

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--config_file",
                        default="bert_config.json",
                        type=str,
                        required=False,
                        help="The BERT model config")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--ckpt_dir",
                        default=None,
                        type=str,
                        help="The ckpt directory, e.g. /results")
    ckpt_group.add_argument("--ckpt_path",
                            default=None,
                            type=str,
                            help="Path to the specific checkpoint")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--eval', dest='do_eval', action='store_true')
    group.add_argument('--prediction', dest='do_eval', action='store_false')
    ## Other parameters
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
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
    parser.add_argument("--ckpt_step",
                        default=-1,
                        type=int,
                        required=False,
                        help="The model checkpoint iteration, e.g. 1000")
                       
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help="Total number of eval  steps to perform, otherwise use full dataset")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--log_path",
                        help="Out file for DLLogger",
                        default="/workspace/dllogger_inference.out",
                        type=str)

    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.log_path),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        assert(args.local_rank != -1) # only use torch.distributed for multi-gpu

    dllogger.log(step="device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16), data={})


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    if args.ckpt_dir:
        if args.ckpt_step == -1:
            #retrieve latest model
            model_names = [f for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")]
            args.ckpt_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
            dllogger.log(step="load model saved at iteration", data={"number": args.ckpt_step})
        model_file = os.path.join(args.ckpt_dir, "ckpt_" + str(args.ckpt_step) + ".pt")
    else:
        model_file = args.ckpt_path
    state_dict = torch.load(model_file, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)

    if args.fp16:
        model.half() # all parameters and buffers are converted to half precision
    model.to(device)

    multi_gpu_training = args.local_rank != -1 and torch.distributed.is_initialized()
    if multi_gpu_training:
        model = DDP(model)
   
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and 'test' in f]
    files.sort()

    dllogger.log(step="***** Running Inference *****", data={})
    dllogger.log(step="  Inference batch", data={"size":args.eval_batch_size})

    model.eval()

    nb_instances = 0
    max_steps = args.max_steps if args.max_steps > 0  else np.inf
    global_step = 0
    total_samples = 0

    begin_infer = time.time()
    with torch.no_grad():
        if args.do_eval:
            final_loss = 0.0 # 
            for data_file in files:
                dllogger.log(step="Opening ", data={"file": data_file})
                dataset = pretraining_dataset(input_file=data_file, max_pred_length=args.max_predictions_per_seq)
                if not multi_gpu_training:
                    train_sampler = RandomSampler(dataset)
                    datasetloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                else:
                    train_sampler = DistributedSampler(dataset)
                    datasetloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                for step, batch in enumerate(tqdm(datasetloader, desc="Iteration")):
                    if global_step > max_steps:
                        break
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch#\
                    loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                    final_loss += loss.item()

                    global_step += 1

                total_samples += len(datasetloader)
                torch.cuda.empty_cache()
                if global_step > max_steps:
                    break
            final_loss /= global_step
            if multi_gpu_training:
                final_loss = torch.tensor(final_loss, device=device)
                dist.all_reduce(final_loss)
                final_loss /= torch.distributed.get_world_size()
            if (not multi_gpu_training or (multi_gpu_training and torch.distributed.get_rank() == 0)):       
                dllogger.log(step="Inference Loss", data={"final_loss": final_loss.item()})


        else: # inference
            # if multi_gpu_training:
            #     torch.distributed.barrier()
            # start_t0 = time.time()
            for data_file in files:
                dllogger.log(step="Opening ", data={"file": data_file})
                dataset = pretraining_dataset(input_file=data_file, max_pred_length=args.max_predictions_per_seq)
                if not multi_gpu_training:
                    train_sampler = RandomSampler(dataset)
                    datasetloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                else:
                    train_sampler = DistributedSampler(dataset)
                    datasetloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

                for step, batch in enumerate(tqdm(datasetloader, desc="Iteration")):
                    if global_step > max_steps:
                        break

                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch#\
                    
                    lm_logits, nsp_logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=None, next_sentence_label=None)

                    nb_instances += input_ids.size(0)
                    global_step += 1

                total_samples += len(datasetloader)
                torch.cuda.empty_cache()
                if global_step > max_steps:
                    break
            # if multi_gpu_training:
            #     torch.distributed.barrier()
            if (not multi_gpu_training or (multi_gpu_training and torch.distributed.get_rank() == 0)):       
                dllogger.log(step="Done Inferring on samples", data={})


    end_infer = time.time()
    dllogger.log(step="Inference perf", data={"inference_sequences_per_second": total_samples * args.eval_batch_size / (end_infer - begin_infer)})


if __name__ == "__main__":
    main()
    dllogger.flush()
