from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import os
import logging
import random
import h5py
from tqdm import tqdm
import datetime
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import json

import unittest

import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from transformers import BertForPreTraining, BertConfig, HfArgumentParser

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

def bert_model_description(config):
    vocab_size = config.vocab_size
    new_model_desc = {
        'inputs': [
            ('input_ids', ['batch', 'max_seq_len_in_batch'],),
            ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
            ('token_type_ids', ['batch', 'max_seq_len_in_batch'],),
            ('masked_lm_labels', ['batch', 'max_seq_len_in_batch'],),
            ('next_sentence_label', ['batch', ],)],
        'outputs': [
            ('loss', [], True),
            ('prediction_scores', ['batch', 'max_seq_len_in_batch', vocab_size],),
            ('seq_relationship_scores', ['batch', 2],)]}
    return new_model_desc


def create_pretraining_dataset(input_file, max_pred_length, args):

    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, num_workers=0,
                                  pin_memory=True)
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

        # HF model use default ignore_index value (-100) for CrossEntropyLoss
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -100
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

@dataclass
class PretrainArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain .hdf5 files  for the task"}
    )

    bert_model: str = field(
        default=None, metadata={"help": "Bert pre-trained model selected in the list: bert-base-uncased, \
            bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."}
    )

    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model checkpoints will be written."}
    )

    cache_dir: str = field(
        default='/tmp/bert_pretrain/',
        metadata={"help": "The output directory where the model checkpoints will be written."}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer \
            than this will be truncated, sequences shorter will be padded."}
    )

    max_predictions_per_seq: Optional[int] = field(
        default=80,
        metadata={"help": "The maximum total of masked tokens in input sequence."}
    )

    train_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for training."}
    )

    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Lamb."}
    )

    num_train_epochs: Optional[float] = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."}
    )

    max_steps: Optional[float] = field(
        default=1000,
        metadata={"help": "Total number of training steps to perform."}
    )

    warmup_proportion: Optional[float] = field(
        default=0.01,
        metadata={"help": "Proportion of training to perform linear learning rate warmup for. \
            E.g., 0.1 = 10%% of training."}
    )

    local_rank: Optional[int] = field(
        default=-1,
        metadata={"help": "local_rank for distributed training on gpus."}
    )

    world_rank: Optional[int] = field(
        default=-1
    )

    world_size: Optional[int] = field(
        default=1
    )

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "random seed for initialization."}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of updates steps to accumualte before performing a backward/update pass."}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit float precision instead of 32-bit."}
    )

    loss_scale: Optional[float] = field(
        default=0.0,
        metadata={"help": "Loss scaling, positive power of 2 values can improve fp16 convergence."}
    )

    log_freq: Optional[float] = field(
        default=1.0,
        metadata={"help": "frequency of logging loss."}
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."}
    )

    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to resume training from checkpoint."}
    )

    resume_step: Optional[int] = field(
        default=-1,
        metadata={"help": "Step to resume training from."}
    )

    num_steps_per_checkpoint: Optional[int] = field(
        default=100,
        metadata={"help": "Number of update steps until a model checkpoint is saved to disk."}
    )

    phase2: bool = field(
        default=False,
        metadata={"help": "Whether to train with seq len 512."}
    )

    allreduce_post_accumulation: bool = field(
        default=False,
        metadata={"help": "Whether to do allreduces during gradient accumulation steps."}
    )

    allreduce_post_accumulation_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to do fp16 allreduce post accumulation."}
    )

    accumulate_into_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 gradient accumulators."}
    )

    phase1_end_step: Optional[int] = field(
        default=7038,
        metadata={"help": "Whether to use fp16 gradient accumulators."}
    )

    tensorboard_dir: Optional[str] = field(
        default=None,
    )

    schedule: Optional[str] = field(
        default='warmup_poly',
    )

    # this argument is test specific. to run a full bert model will take too long to run. instead, we reduce
    # number of hidden layers so that it can show convergence to an extend to help detect any regression.
    force_num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Whether to use fp16 gradient accumulators."}
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        args.local_rank = 0

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

    logger.info("setup_training: args.train_batch_size = %d", args.train_batch_size)
    return device, args


def prepare_model(args, device):
    config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
    
    # config.num_hidden_layers = 12
    if args.force_num_hidden_layers:
        logger.info("Modifying model config with num_hidden_layers to %d", args.force_num_hidden_layers)
        config.num_hidden_layers = args.force_num_hidden_layers

    model = BertForPreTraining(config)
    model_desc = bert_model_description(config)

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

    optim_config = optim.AdamConfig(params=params, lr=2e-5, do_bias_correction=True)
    model = orttrainer.ORTTrainer(model, model_desc, optim_config, options=options)

    return model

def get_data_file(f_id, world_rank, world_size, files):
    num_files = len(files)
    if world_size > num_files:
        remainder = world_size % num_files
        return files[(f_id * world_size + world_rank + remainder * f_id) % num_files]
    elif world_size > 1:
        return files[(f_id * world_size + world_rank) % num_files]
    else:
        return files[f_id % num_files]


def main():
    parser = HfArgumentParser(PretrainArguments)
    args = parser.parse_args_into_dataclasses()[0]
    do_pretrain(args)


def do_pretrain(args):
    if is_main_process(args) and args.tensorboard_dir:
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        tb_writer.add_text("args", args.to_json_string())
        tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})
    else:
        tb_writer = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ort.set_seed(args.seed)

    device, args = setup_training(args)

    model = prepare_model(args, device)

    logger.info("Running training: Batch size = %d, initial LR = %f", args.train_batch_size, args.learning_rate)

    most_recent_ckpts_paths = []
    average_loss = 0.0
    epoch = 0
    training_steps = 0

    pool = ProcessPoolExecutor(1)
    while True:
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
        files.sort()
        random.shuffle(files)

        f_id = 0
        train_dataloader, data_file = create_pretraining_dataset(
            get_data_file(f_id, args.world_rank, args.world_size, files),
            args.max_predictions_per_seq,
            args)

        for f_id in range(1 , len(files)):
            logger.info("data file %s" % (data_file))

            dataset_future = pool.submit(
                create_pretraining_dataset,
                get_data_file(f_id, args.world_rank, args.world_size, files),
                args.max_predictions_per_seq,
                args)

            train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process(args) else train_dataloader
            for step, batch in enumerate(train_iter):
                training_steps += 1
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

                loss, _, _ = model.train_step(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
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

            train_dataloader, data_file = dataset_future.result(timeout=None)

        epoch += 1


def generate_tensorboard_logdir(root_dir): 
    current_date_time = datetime.datetime.today()

    dt_string = current_date_time.strftime('BERT_pretrain_%y_%m_%d_%I_%M_%S')
    return os.path.join(root_dir, dt_string)


class ORTBertPretrainTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = '/bert_data/hf_data/test_out/bert_pretrain_results'
        self.bert_model = 'bert-base-uncased'
        self.local_rank = -1
        self.world_rank = -1
        self.world_size = 1
        self.max_steps = 300000
        self.learning_rate = 5e-4
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.input_dir = '/bert_data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/train'
        self.train_batch_size = 4096
        self.gradient_accumulation_steps = 64
        self.fp16 = True
        self.allreduce_post_accumulation = True
        self.tensorboard_dir = '/bert_data/hf_data/test_out'

    def test_pretrain_throughput(self):
        # setting train_batch_size and gradient_accumulation_steps to maximize per gpu memory usage under 16GB
        # to validate throughput regression.
        # train_batch_size is initially configured as per optimization batch size,
        # taking into consideration of world_size and gradient_accumulation_steps:
        # train_batch_size = world_size * gradient_accumulation_steps * batch_size_per_gpu
        # in the code later train_batch_size is translated to per gpu batch size:
        # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps // args.world_size

        # the LAMB batch size of 64k
        optimization_batch_size = 64 * 1024
        per_gpu_batch_size = 32

        self.train_batch_size = optimization_batch_size
        self.gradient_accumulation_steps = optimization_batch_size // per_gpu_batch_size // self.world_size

        logger.info("self.gradient_accumulation_steps = %d", self.gradient_accumulation_steps)

        # only to run on  optimization step because we only want to make sure there is no throughput regression
        self.max_steps = 1
        args = PretrainArguments(
            output_dir=self.output_dir,
            bert_model=self.bert_model,
            local_rank=self.local_rank,
            world_rank=self.world_rank,
            world_size=self.world_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            max_seq_length=self.max_seq_length,
            max_predictions_per_seq=self.max_predictions_per_seq,
            train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            input_dir=self.input_dir,
            fp16=self.fp16,
            allreduce_post_accumulation=self.allreduce_post_accumulation)
        do_pretrain(args)

    def test_pretrain_convergence(self):
        args = PretrainArguments(
            output_dir=self.output_dir,
            bert_model=self.bert_model,
            local_rank=self.local_rank,
            world_rank=self.world_rank,
            world_size=self.world_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            max_seq_length=self.max_seq_length,
            max_predictions_per_seq=self.max_predictions_per_seq,
            train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            input_dir=self.input_dir,
            fp16=self.fp16,
            allreduce_post_accumulation=self.allreduce_post_accumulation,
            force_num_hidden_layers=self.force_num_hidden_layers,
            tensorboard_dir=generate_tensorboard_logdir('/bert_data/hf_data/test_out/'))
        final_loss = do_pretrain(args)
        return final_loss


# to do parallel training:
# python -m torch.distributed.launch --nproc_per_node 4 orttraining_run_bert_pretrain.py
if __name__ == "__main__":
    import sys
    logger.warning("sys.argv: %s", sys.argv)
    # usage:
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py ORTBertPretrainTest.test_pretrain_throughput
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py ORTBertPretrainTest.test_pretrain_convergence
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py     # to run real BERT convergence test
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

        test = ORTBertPretrainTest()
        test.setUp()
        test.local_rank = local_rank
        test.world_rank = local_rank
        test.world_size = world_size

        if len(sys.argv) >= 2 and sys.argv[1] == 'ORTBertPretrainTest.test_pretrain_throughput':
            logger.info("running ORTBertPretrainTest.test_pretrain_throughput()...")
            test.test_pretrain_throughput()
            logger.info("ORTBertPretrainTest.test_pretrain_throughput() passed")
        elif len(sys.argv) >= 2 and sys.argv[1] == 'ORTBertPretrainTest.test_pretrain_convergence':
            logger.info("running ORTBertPretrainTest.test_pretrain_convergence()...")
            test.max_steps = 200
            test.force_num_hidden_layers = 8
            final_loss = test.test_pretrain_convergence()
            logger.info("ORTBertPretrainTest.test_pretrain_convergence() final loss = %f", final_loss)
            test.assertLess(final_loss, 8.5)
            logger.info("ORTBertPretrainTest.test_pretrain_convergence() passed")
        else:
            # https://microsoft.sharepoint.com/teams/ONNX2/_layouts/15/Doc.aspx?sourcedoc={170774be-e1c6-4f8b-a3ae-984f211fe410}&action=edit&wd=target%28ONNX%20Training.one%7C8176133b-c7cb-4ef2-aa9d-3fdad5344c40%2FGitHub%20Master%20Merge%20Schedule%7Cb67f0db1-e3a0-4add-80a6-621d67fd8107%2F%29
            # to make equivalent args for cpp convergence test

            # ngpu=4 
            # seq_len=128 
            # max_predictions_per_seq=20 
            # batch_size=64 
            # grad_acc=16 
            # num_train_steps=1000000 
            # optimizer=adam
            # lr=5e-4 
            # warmup_ratio=0.1 
            # warmup_mode=Linear 
            # effective_batch_size=$((ngpu * batch_size * grad_acc)) 
            # commit=$(git rev-parse HEAD | cut -c1-8) 
            # time_now=$(date +%m%d%H%M) 
            # run_name=ort_${commit}_nvbertbase_bookwiki128_fp16_${optimizer}_lr${lr}_${warmup_mode}${warmup_ratio}_g${ngpu}_bs${batch_size}_acc${grad_acc}_efbs${effective_batch_size}_steps${num_train_steps}_fp16allreduce_${time_now} 

            # mixed precision 
            # mpirun -n ${ngpu} ./onnxruntime_training_bert --model_name /bert_ort/bert_models/nv/bert-base/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm
            #   --train_data_dir /bert_data/128/books_wiki_en_corpus/train --test_data_dir /bert_data/128/books_wiki_en_corpus/test
            #   --train_batch_size ${batch_size} --mode train --num_train_steps ${num_train_steps} --display_loss_steps 5 
            #   --log_dir ./logs/bert_base/${run_name} --optimizer ${optimizer} --learning_rate ${lr} --warmup_ratio ${warmup_ratio} --warmup_mode ${warmup_mode} 
            #   --gradient_accumulation_steps ${grad_acc} --max_predictions_per_seq=${max_predictions_per_seq} --use_mixed_precision --allreduce_in_fp16 --lambda 0
            #   --use_nccl | tee ${run_name}.log 

            test.max_seq_length = 128
            test.max_predictions_per_seq = 20
            test.gradient_accumulation_steps = 16
            test.train_batch_size = 64 * test.gradient_accumulation_steps * test.world_size    # cpp_batch_size (=64) * grad_acc * world_size
            test.max_steps = 300000

            test.force_num_hidden_layers = None

            # already using Adam (e.g. AdamConfig)
            test.learning_rate = 5e-4
            test.warmup_proportion = 0.1

            final_loss = test.test_pretrain_convergence()
            logger.info("ORTBertPretrainTest.test_pretrain_convergence() final loss = %f", final_loss)
    else:
        unittest.main()
