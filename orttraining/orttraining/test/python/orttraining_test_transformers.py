from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import sys
import os
from enum import Enum

# set transformer_repo to the huggingface repo location
transformer_repo = ''
nvidia_deep_learning_examples_repo = ''

sys.path.append(transformer_repo)

# from configuration_common_test import ConfigTester

# from modeling_common_test import (CommonTestCases, ids_tensor, floats_tensor)

import random
global_rng = random.Random()

def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


from transformers import (BertConfig, BertModel, BertForMaskedLM,
                            BertForNextSentencePrediction, BertForPreTraining,
                            BertForQuestionAnswering, BertForSequenceClassification,
                            BertForTokenClassification, BertForMultipleChoice)

sys.path.append("/bert_ort/liqun/onnxruntime/build/Linux/Debug/")
import onnxruntime as ort
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample
import torch

from helpers.bert_postprocess import postprocess_model

# sys.path.append(os.path.join(nvidia_deep_learning_examples_repo, 'PyTorch/LanguageModeling/BERT'))
# from run_pretraining import postprocess_model


def map_optimizer_attributes(name):
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay = any(no_decay_key in name for no_decay_key in no_decay_keys)
    if no_decay:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
    else:
        return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

from torch.utils.data import Dataset, DataLoader
class OrtTestDataset(Dataset):
    def __init__(self, input_desc, seq_len, device):
        import copy
        self.input_desc_ = copy.deepcopy(input_desc)
        for input_desc in self.input_desc_:
            shape_ = []
            for i, axis in enumerate(input_desc.shape_):
                if axis == 'max_seq_len_in_batch':
                    shape_ = shape_ + [seq_len, ]
                elif axis != 'batch':
                    shape_ = input_desc.shape_[i]
            input_desc.shape_ = shape_
        self.device_ = device

    def __len__(self):
        return 100

    def __getitem__(self, item):
        input_batch = []
        for input_desc in self.input_desc_:
            input_sample = generate_sample(input_desc, self.device_)
            input_batch.append(input_sample)
        return input_batch

def create_ort_test_dataloader(input_desc, batch_size, seq_len, device):
    dataset = OrtTestDataset(input_desc, seq_len, device)
    return DataLoader(dataset, batch_size=batch_size)

class BatchArgsOption(Enum):
    List = 1
    Dict = 2
    ListAndDict = 3

def split_batch(batch, input_desc, args_count):
    total_argument_count = len(input_desc)
                # batch=[input_ids[batch, seglen], attention_mask[batch, seglen], token_type_ids[batch,seglen], token_type_ids[batch, seglen]]
    args = []   # (input_ids[batch, seglen], attention_mask[batch, seglen])
    kwargs = {} # {'token_type_ids': token_type_ids[batch,seglen], 'position_ids': token_type_ids[batch, seglen]}
    for i in range(args_count):
        args = args + [batch[i]]

    for i in range(args_count, total_argument_count):
        kwargs[input_desc[i].name_] = batch[i]

    return args, kwargs

def run_test(model, model_desc, device, args, gradient_accumulation_steps, fp16,
    allreduce_post_accumulation, get_lr_this_step, use_internal_get_lr_this_step, loss_scaler, use_internal_loss_scaler, 
    batch_args_option):
    dataloader = create_ort_test_dataloader(model_desc.inputs_, args.batch_size, args.seq_len, device)

    model = ORTTrainer(model, None, model_desc, "LambOptimizer",
        map_optimizer_attributes=map_optimizer_attributes,
        learning_rate_description=IODescription('Learning_Rate', [1,], torch.float32),
        device=device, postprocess_model=postprocess_model,
        gradient_accumulation_steps=gradient_accumulation_steps,                
        # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
        world_rank=args.local_rank, world_size=args.world_size,
        use_mixed_precision=fp16,
        allreduce_post_accumulation=allreduce_post_accumulation,
        get_lr_this_step=get_lr_this_step if use_internal_get_lr_this_step else None,
        loss_scaler=loss_scaler if use_internal_loss_scaler else None)

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


class BertModelTest(unittest.TestCase):

    class BertModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     use_labels=True,
                     vocab_size=99,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     scope=None,
                     device='cpu',
                     ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope
            self.device = device

            # 1. superset of bert input/output descs
            # see BertPreTrainedModel doc
            self.input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.vocab_size)
            self.attention_mask_desc = IODescription('attention_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
            self.token_type_ids_desc = IODescription('token_type_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
            self.position_ids_desc = IODescription('position_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.max_position_embeddings)
            self.head_mask_desc = IODescription('head_mask', [self.num_hidden_layers, self.num_attention_heads], torch.int64, num_classes=2)
            self.inputs_embeds_desc = IODescription('inputs_embeds', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)

            self.encoder_hidden_states_desc = IODescription('encoder_hidden_states', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.encoder_attention_mask_desc = IODescription('encoder_attention_mask', ['batch', 'max_seq_len_in_batch'], torch.float32)

            # see BertForPreTraining doc
            self.masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=self.vocab_size)
            self.next_sentence_label_desc = IODescription('next_sentence_label', ['batch',], torch.int64, num_classes=2)

            # outputs
            self.loss_desc = IODescription('loss', [1,], torch.float32)
            self.prediction_scores_desc = IODescription('prediction_scores', ['batch', 'max_seq_len_in_batch', self.vocab_size], torch.float32)

            self.seq_relationship_scores_desc = IODescription('seq_relationship_scores', ['batch', 2], torch.float32)   # IODescription('seq_relationship_scores', ['batch', 'max_seq_len_in_batch', 2], torch.float32)
            self.hidden_states_desc = IODescription('hidden_states', [self.num_hidden_layers, 'batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.attentions_desc = IODescription('attentions', [self.num_hidden_layers, 'batch', self.num_attention_heads, 'max_seq_len_in_batch', 'max_seq_len_in_batch'], torch.float32)
            self.last_hidden_state_desc = IODescription('last_hidden_state', ['batch', 'max_seq_len_in_batch', self.hidden_size], torch.float32)
            self.pooler_output_desc = IODescription('pooler_output', ['batch', self.hidden_size], torch.float32)

        def BertForPreTraining_descs(self):
            return ModelDescription(
                [self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc, self.masked_lm_labels_desc, self.next_sentence_label_desc],
                # returns loss_desc if both masked_lm_labels_desc, next_sentence_label are provided
                # hidden_states_desc, attentions_desc shall be included according to config.output_attentions, config.output_hidden_states
                [self.loss_desc, self.prediction_scores_desc, self.seq_relationship_scores_desc, 
                #hidden_states_desc, attentions_desc
                ])

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(self.device)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2).to(self.device)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size).to(self.device)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size).to(self.device)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels).to(self.device)
                choice_labels = ids_tensor([self.batch_size], self.num_choices).to(self.device)

            config = BertConfig(
                vocab_size=self.vocab_size,
                vocab_size_or_config_json_file=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range)

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def create_and_check_bert_for_pretraining(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = BertForPreTraining(config=config)
            model.eval()
            loss, prediction_scores, seq_relationship_score = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                                                    masked_lm_labels=token_labels, next_sentence_label=sequence_labels)
            model_desc = ModelDescription([self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc,
                                           self.masked_lm_labels_desc, self.next_sentence_label_desc],
                                          [self.loss_desc, self.prediction_scores_desc, self.seq_relationship_scores_desc])

            import argparse
            args_ = argparse.Namespace(fp16=True, amp_opt_level='O1')

            from collections import namedtuple
            MyArgs = namedtuple("MyArgs", 
                "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len")
            args = MyArgs(local_rank=0, world_size=1, max_steps=100, learning_rate=0.00001, warmup_proportion=0.01, batch_size=13, seq_len=7)

            from helpers.utils import get_lr
            def get_lr_this_step(global_step):
                return get_lr(args, global_step)
            loss_scaler = LossScaler('loss_scale_input_name', True, up_scale_window=2000)

            option_gradient_accumulation_steps = [8]
            option_fp16 = [True]
            option_allreduce_post_accumulation = True
            option_use_internal_get_lr_this_step = False
            option_use_internal_loss_scaler = True
            option_split_batch = [BatchArgsOption.ListAndDict]
            # TODO: with with fetches

            for gradient_accumulation_steps in option_gradient_accumulation_steps:
                for fp16 in option_fp16:
                    for split_batch in option_split_batch:                
                        loss_ort, prediction_scores_ort, seq_relationship_score_ort =\
                            run_test(model, model_desc, self.device, args, gradient_accumulation_steps, fp16,
                                     option_allreduce_post_accumulation,
                                     get_lr_this_step, option_use_internal_get_lr_this_step,
                                     loss_scaler, option_use_internal_loss_scaler,
                                     split_batch)

                        print(loss_ort)
                        print(prediction_scores_ort)
                        print(seq_relationship_score_ort)

    def setUp(self):
        self.model_tester = BertModelTest.BertModelTester(self)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(*config_and_inputs)

if __name__ == "__main__":
    unittest.main()
