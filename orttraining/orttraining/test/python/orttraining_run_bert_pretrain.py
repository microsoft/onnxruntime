from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import os
import logging
import argparse
import random
import h5py
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset

from transformers import BertForPreTraining, BertConfig

from modeling import BertForPreTraining as NV_BertForPreTraining, BertConfig as NV_BertConfig

from utils import is_main_process

from concurrent.futures import ProcessPoolExecutor

import onnxruntime as ort
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription
from onnxruntime.capi.ort_trainer import LossScaler

from onnxruntime.experimental import amp, optim, orttrainer
from onnxruntime.experimental.optim import _LRScheduler, PolyWarmupLRScheduler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def bert_model_description(config):
    vocab_size = config.vocab_size
    input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    attention_mask_desc = IODescription('attention_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    token_type_ids_desc = IODescription('token_type_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    next_sentence_label_desc = IODescription('next_sentence_label', ['batch',], torch.int64, num_classes = 2)
    loss_desc = IODescription('loss', [], torch.float32)
    prediction_scores_desc = IODescription('prediction_scores', ['batch', 'max_seq_len_in_batch', vocab_size], torch.float32)
    seq_relationship_scores_desc = IODescription('seq_relationship_scores', ['batch', 2], torch.float32)
    return ModelDescription(
        [input_ids_desc, attention_mask_desc, token_type_ids_desc, masked_lm_labels_desc, next_sentence_label_desc],
        [loss_desc, prediction_scores_desc, seq_relationship_scores_desc])


def new_bert_model_description(config):
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
    parser.add_argument('--accumulate_into_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use fp16 gradient accumulators.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--tensorboard_dir',
                        default=None,
                        type=str)
    parser.add_argument('--use_ort_trainer',
                        default=False,
                        action='store_true',
                        help="Whether to run with ort in fully optimized mode (run optimization in ort as opposed in pytorch).")
    parser.add_argument('--schedule',
                        default='warmup_poly',
                        type=str)
    parser.add_argument('--use_ort_trainer_nccl',
                        default=False,
                        action='store_true',
                        help="Whether to run with ort trainer with NCCL instead of Horovod.")
    parser.add_argument('--use_ib',
                        default=False,
                        help='Whether to enable IB or not')                    
    args = parser.parse_args()
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.n_gpu = 1

    from onnxruntime.capi._pybind_state import set_cuda_device_id 
    set_cuda_device_id(args.local_rank)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    return device, args

def prepare_model(args, device):
    # config = NV_BertConfig.from_json_file(args.config_file)
    # if config.vocab_size % 8 != 0:
    #     config.vocab_size += 8 - (config.vocab_size % 8)
    # model = NV_BertForPreTraining(config)

    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForPreTraining(config)
    model_desc = bert_model_description(config)

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

    from train_with_ort_trainer import get_lr

    def get_lr_this_step(global_step):
        return get_lr(args, global_step, args.schedule)
    loss_scaler = LossScaler('loss_scale_input_name', True, up_scale_window=2000) if args.fp16 else None

    model = ORTTrainer(
        model,
        None,
        model_desc,
        "LambOptimizer",
        map_optimizer_attributes=map_optimizer_attributes,
        learning_rate_description=IODescription('Learning_Rate', [1,], torch.float32),
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=args.fp16,
        allreduce_post_accumulation=args.allreduce_post_accumulation,
        get_lr_this_step=get_lr_this_step,
        loss_scaler=loss_scaler,
        _opset_version=12)

    return model

def new_prepare_model(args, device):
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForPreTraining(config)
    model_desc = new_bert_model_description(config)

    from train_with_ort_trainer import get_lr

    # class WrapLRScheduler(_LRScheduler):
    #     def __init__(self, args, args_schedule):
    #         super().__init__()
    #         self.args_ = args
    #         self.args_schedule_ = args_schedule

    #     def get_lr(self, train_step_info):
    #         return [get_lr(args, train_step_info.optimization_step, args.schedule)]

    # lr_scheduler = WrapLRScheduler(args, args.schedule)
    lr_scheduler = PolyWarmupLRScheduler(total_steps=int(args.max_steps))

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
                                                'allreduce_post_accumulation': True},
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


def main():

    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ort.set_seed(args.seed)

    device, args = setup_training(args)

    model = new_prepare_model(args, device)

    logger.info("Running training: Batch size = %d, initial LR = %f", args.train_batch_size, args.learning_rate)

    global_step = 0
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

        train_dataloader, data_file = create_pretraining_dataset(files[0], args.max_predictions_per_seq, args)

        for f_id in range(1 , len(files)):
            logger.info("data file %s" % (data_file))

            dataset_future = pool.submit(create_pretraining_dataset, files[f_id], args.max_predictions_per_seq, args)

            train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process(args) else train_dataloader
            for step, batch in enumerate(train_iter):
                training_steps += 1
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

                loss, _, _ = model.train_step(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)

                global_step += 1

                average_loss += loss.item()

                if global_step >= args.max_steps:
                    last_num_steps = global_step % args.log_freq
                    last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                    average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                    average_loss = average_loss / (last_num_steps * args.gradient_accumulation_steps)
                    logger.info("Total Steps:{} Final Loss = {}".format(training_steps, average_loss.item()))
                    return 
                elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    if is_main_process(args):
                        print("Step:{} Average Loss = {}".format(global_step, average_loss / (
                                    args.log_freq * args.gradient_accumulation_steps)))
                    average_loss = 0

            del train_dataloader
            # Make sure pool has finished and switch train_dataloader
            # NOTE: Will block until complete
            train_dataloader, data_file = dataset_future.result(timeout=None)

        epoch += 1


if __name__ == "__main__":
    main()
