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


import argparse

# ==================
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import amp_C
import apex_C
import h5py
import numpy as np
import torch
from apex import amp
from apex.amp import _amp_state
from apex.parallel import DistributedDataParallel as DDP  # noqa: N817
from apex.parallel.distributed import flat_dist_call
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE  # noqa: F401
from modeling import BertConfig, BertForPreTraining
from optimization import BertLAMB
from schedulers import LinearWarmUpScheduler  # noqa: F401
from tokenization import BertTokenizer  # noqa: F401
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler  # noqa: F401
from torch.utils.data.distributed import DistributedSampler  # noqa: F401
from tqdm import tqdm, trange  # noqa: F401
from utils import is_main_process

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size * args.n_gpu, num_workers=4, pin_memory=True
    )
    # shared_list["0"] = (train_dataloader, input_file)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):  # noqa: N801
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            (
                torch.from_numpy(input[index].astype(np.int64))
                if indice < 5
                else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            )
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain .hdf5 files  for the task.",
    )

    parser.add_argument("--config_file", default=None, type=str, required=True, help="The BERT model config")

    parser.add_argument(
        "--bert_model",
        default="bert-large-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--max_predictions_per_seq", default=80, type=int, help="The maximum total of masked tokens in input sequence"
    )
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--max_steps", default=1000, type=float, help="Total number of training steps to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16", default=False, action="store_true", help="Whether to use 16-bit float precision instead of 32-bit"
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0.0,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument("--log_freq", type=float, default=50.0, help="frequency of logging loss.")
    parser.add_argument(
        "--checkpoint_activations", default=False, action="store_true", help="Whether to use gradient checkpointing"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="Whether to resume training from checkpoint.",
    )
    parser.add_argument("--resume_step", type=int, default=-1, help="Step to resume training from.")
    parser.add_argument(
        "--num_steps_per_checkpoint",
        type=int,
        default=100,
        help="Number of update steps until a model checkpoint is saved to disk.",
    )
    parser.add_argument("--phase2", default=False, action="store_true", help="Whether to train with seq len 512")
    parser.add_argument(
        "--allreduce_post_accumulation",
        default=False,
        action="store_true",
        help="Whether to do allreduces during gradient accumulation steps.",
    )
    parser.add_argument(
        "--allreduce_post_accumulation_fp16",
        default=False,
        action="store_true",
        help="Whether to do fp16 allreduce post accumulation.",
    )
    parser.add_argument(
        "--accumulate_into_fp16", default=False, action="store_true", help="Whether to use fp16 gradient accumulators."
    )
    parser.add_argument(
        "--phase1_end_step", type=int, default=7038, help="Number of training steps in Phase1 - seq len 128"
    )
    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    args = parser.parse_args()
    return args


def setup_training(args):
    assert torch.cuda.is_available()

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger.info("device %s n_gpu %d distributed training %r", device, args.n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, batch size {args.train_batch_size} should be divisible"
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if (
        not args.resume_from_checkpoint
        and os.path.exists(args.output_dir)
        and (os.listdir(args.output_dir) and os.listdir(args.output_dir) != ["logfile.txt"])
    ):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def prepare_model_and_optimizer(args, device):
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split(".pt")[0].split("_")[1].strip()) for x in model_names])
        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, f"ckpt_{global_step}.pt"), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.phase2:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm"]

    optimizer_grouped_parameters = []
    names = []

    count = 1
    for n, p in param_optimizer:
        count += 1
        if not any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({"params": [p], "weight_decay": 0.01, "name": n})
            names.append({"params": [n], "weight_decay": 0.01})
        if any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({"params": [p], "weight_decay": 0.00, "name": n})
            names.append({"params": [n], "weight_decay": 0.00})

    optimizer = BertLAMB(
        optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=args.max_steps
    )
    if args.fp16:
        if args.loss_scale == 0:
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level="O2",
                loss_scale="dynamic",
                master_weights=not args.accumulate_into_fp16,
            )
        else:
            # optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level="O2",
                loss_scale=args.loss_scale,
                master_weights=not args.accumulate_into_fp16,
            )
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.resume_from_checkpoint:
        if args.phase2:
            keys = list(checkpoint["optimizer"]["state"].keys())
            # Override hyperparameters from Phase 1
            for key in keys:
                checkpoint["optimizer"]["state"][key]["step"] = global_step
            for iter, _item in enumerate(checkpoint["optimizer"]["param_groups"]):
                checkpoint["optimizer"]["param_groups"][iter]["t_total"] = args.max_steps
                checkpoint["optimizer"]["param_groups"][iter]["warmup"] = args.warmup_proportion
                checkpoint["optimizer"]["param_groups"][iter]["lr"] = args.learning_rate
        optimizer.load_state_dict(checkpoint["optimizer"])  # , strict=False)

        # Restore AMP master parameters
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint["optimizer"])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint["master params"]):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, checkpoint, global_step


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device="cuda", dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (torch.distributed.get_world_size() * args.gradient_accumulation_steps),
        )
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536, overflow_buf, [allreduced_views, master_grads], 1.0 / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            if is_main_process():
                print(
                    f"Rank {torch.distributed.get_rank()} :: Gradient overflow.  Skipping step, reducing loss scale to {scaler.loss_scale()}"
                )
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, args = setup_training(args)

    # Prepare optimizer
    config = BertConfig.from_json_file(args.config_file)
    model, optimizer, checkpoint, global_step = prepare_model_and_optimizer(args, device)
    is_model_exported = False

    if is_main_process():
        print(f"SEED {args.seed}")

    if args.do_train:
        if is_main_process():
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", args.train_batch_size)
            print("  LR = ", args.learning_rate)
            print("Training. . .")

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0

        pool = ProcessPoolExecutor(1)

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            if not args.resume_from_checkpoint or epoch > 0 or args.phase2:
                files = [
                    os.path.join(args.input_dir, f)
                    for f in os.listdir(args.input_dir)
                    if os.path.isfile(os.path.join(args.input_dir, f))
                ]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint["files"][0]
                files = checkpoint["files"][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)
            print("File list is [" + ",".join(files) + "].")

            shared_file_list = {}

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                remainder = torch.distributed.get_world_size() % num_files
                data_file = files[
                    (
                        f_start_id * torch.distributed.get_world_size()
                        + torch.distributed.get_rank()
                        + remainder * f_start_id
                    )
                    % num_files
                ]
            else:
                data_file = files[f_start_id % num_files]

            previous_file = data_file

            print(f"Create pretraining_dataset with file {data_file}...")
            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(
                train_data,
                sampler=train_sampler,
                batch_size=args.train_batch_size * args.n_gpu,
                num_workers=4,
                pin_memory=True,
            )
            # shared_file_list["0"] = (train_dataloader, data_file)

            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])

            for f_id in range(f_start_id + 1, len(files)):
                # torch.cuda.synchronize()
                # f_start = time.time()
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
                    data_file = files[
                        (f_id * torch.distributed.get_world_size() + torch.distributed.get_rank() + remainder * f_id)
                        % num_files
                    ]
                else:
                    data_file = files[f_id % num_files]

                logger.info(f"file no {f_id} file {previous_file}")

                previous_file = data_file

                # train_dataloader = shared_file_list["0"][0]

                # thread = multiprocessing.Process(
                #     name="LOAD DATA:" + str(f_id) + ":" + str(data_file),
                #     target=create_pretraining_dataset,
                #     args=(data_file, args.max_predictions_per_seq, shared_file_list, args, n_gpu)
                # )
                # thread.start()
                print(f"Submit new data file {data_file} for the next iteration...")
                dataset_future = pool.submit(
                    create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args
                )
                # torch.cuda.synchronize()
                # f_end = time.time()
                # print('[{}] : shard overhead {}'.format(torch.distributed.get_rank(), f_end - f_start))

                train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
                for _step, batch in enumerate(train_iter):
                    # torch.cuda.synchronize()
                    # iter_start = time.time()

                    training_steps += 1
                    batch = [t.to(device) for t in batch]  # noqa: PLW2901
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    if not is_model_exported:
                        onnx_path = os.path.join(
                            args.output_dir, "bert_for_pretraining_without_loss_" + config.to_string() + ".onnx"
                        )
                        lm_score, sq_score = model(
                            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                        )
                        torch.onnx.export(
                            model,
                            (input_ids, segment_ids, input_mask),
                            onnx_path,
                            verbose=True,
                            # input_names = ['input_ids', 'token_type_ids', 'input_mask'],
                            input_names=["input1", "input2", "input3"],
                            output_names=["output1", "output2"],
                            dynamic_axes={
                                "input1": {0: "batch"},
                                "input2": {0: "batch"},
                                "input3": {0: "batch"},
                                "output1": {0: "batch"},
                                "output2": {0: "batch"},
                            },
                            training=True,
                        )
                        is_model_exported = False

                        import onnxruntime as ort

                        sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
                        result = sess.run(
                            None,
                            {
                                "input1": input_ids.cpu().numpy(),
                                "input2": segment_ids.cpu().numpy(),
                                "input3": input_mask.cpu().numpy(),
                            },
                        )

                        print("---ORT result---")
                        print(result[0])
                        print(result[1])

                        print("---Pytorch result---")
                        print(lm_score)
                        print(sq_score)

                        print("---ORT-Pytorch Diff---")
                        print(np.linalg.norm(result[0] - lm_score.detach().cpu().numpy()))
                        print(np.linalg.norm(result[1] - sq_score.detach().cpu().numpy()))
                        return

                    loss = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        masked_lm_labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels,
                        checkpoint_activations=args.checkpoint_activations,
                    )
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    divisor = args.gradient_accumulation_steps
                    if args.gradient_accumulation_steps > 1:
                        if not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / args.gradient_accumulation_steps
                            divisor = 1.0
                    if args.fp16:
                        with amp.scale_loss(
                            loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation
                        ) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    average_loss += loss.item()

                    if training_steps % args.gradient_accumulation_steps == 0:
                        global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)

                    if global_step >= args.max_steps:
                        last_num_steps = global_step % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if torch.distributed.is_initialized():
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        if is_main_process():
                            logger.info(f"Total Steps:{training_steps} Final Loss = {average_loss.item()}")
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if is_main_process():
                            print(
                                "Step:{} Average Loss = {} Step Loss = {} LR {}".format(
                                    global_step,
                                    average_loss / (args.log_freq * divisor),
                                    loss.item() * args.gradient_accumulation_steps / divisor,
                                    optimizer.param_groups[0]["lr"],
                                )
                            )
                        average_loss = 0

                    if (
                        global_step >= args.max_steps
                        or training_steps % (args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0
                    ):
                        if is_main_process():
                            # Save a trained model
                            logger.info("** ** * Saving fine - tuned model ** ** * ")
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, f"ckpt_{global_step}.pt")
                            else:
                                output_save_file = os.path.join(
                                    args.output_dir, f"ckpt_{global_step + args.phase1_end_step}.pt"
                                )
                            if args.do_train:
                                torch.save(
                                    {
                                        "model": model_to_save.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "master params": list(amp.master_params(optimizer)),
                                        "files": [f_id, *files],
                                    },
                                    output_save_file,
                                )

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps:
                            del train_dataloader
                            # thread.join()
                            return args

                    # torch.cuda.synchronize()
                    # iter_end = time.time()

                    # if torch.distributed.get_rank() == 0:
                    #     print('step {} : {}'.format(global_step, iter_end - iter_start))

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1


if __name__ == "__main__":
    now = time.time()
    args = main()
    if is_main_process():
        print(f"Total time taken {time.time() - now}")
