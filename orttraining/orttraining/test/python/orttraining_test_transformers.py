from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import os
import random
import numpy as np
from transformers import (BertConfig, BertForPreTraining, BertModel)

from orttraining_test_data_loader import ids_tensor, BatchArgsOption
from orttraining_test_utils import run_test, get_lr

import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler

import torch

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

        def create_and_check_bert_for_pretraining(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch,
            option_use_internal_get_lr_this_step=[True],
            option_use_internal_loss_scaler=[True]):
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            onnxruntime.set_seed(seed)

            model = BertForPreTraining(config=config)
            model.eval()
            loss, prediction_scores, seq_relationship_score = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                                                    masked_lm_labels=token_labels, next_sentence_label=sequence_labels)
            model_desc = ModelDescription([self.input_ids_desc, self.attention_mask_desc, self.token_type_ids_desc,
                                           self.masked_lm_labels_desc, self.next_sentence_label_desc],
                                          [self.loss_desc, self.prediction_scores_desc, self.seq_relationship_scores_desc])

            from collections import namedtuple
            MyArgs = namedtuple("MyArgs",
                "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len")
            args = MyArgs(local_rank=0, world_size=1, max_steps=100, learning_rate=0.00001, warmup_proportion=0.01, batch_size=13, seq_len=7)

            def get_lr_this_step(global_step):
                return get_lr(args, global_step)
            loss_scaler = LossScaler('loss_scale_input_name', True, up_scale_window=2000)

            for fp16 in option_fp16:
                for allreduce_post_accumulation in option_allreduce_post_accumulation:
                    for gradient_accumulation_steps in option_gradient_accumulation_steps:
                        for use_internal_get_lr_this_step in option_use_internal_get_lr_this_step:
                            for use_internal_loss_scaler in option_use_internal_loss_scaler:
                                for split_batch in option_split_batch:
                                    print("gradient_accumulation_steps:", gradient_accumulation_steps)
                                    print("split_batch:", split_batch)
                                    loss_ort, prediction_scores_ort, seq_relationship_score_ort =\
                                        run_test(
                                            model, model_desc, self.device, args, gradient_accumulation_steps, fp16,
                                            allreduce_post_accumulation,
                                            get_lr_this_step, use_internal_get_lr_this_step,
                                            loss_scaler, use_internal_loss_scaler,
                                            split_batch)

                                    print(loss_ort)
                                    print(prediction_scores_ort)
                                    print(seq_relationship_score_ort)

    def setUp(self):
        self.model_tester = BertModelTest.BertModelTester(self)

    def test_for_pretraining_mixed_precision_all(self):
        # It would be better to test both with/without mixed precision and allreduce_post_accumulation.
        # However, stress test of all the 4 cases is not stable at least on the test machine.
        # There we only test mixed precision and allreduce_post_accumulation because it is the most useful use cases.
        option_fp16 = [True]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [1, 8]
        option_split_batch = [BatchArgsOption.ListAndDict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_all(self):
        # This test is not stable because it create and run ORTSession multiple times.
        # It occasionally gets seg fault at ~MemoryPattern()
        # when releasing patterns_. In order not to block PR merging CI test,
        # this test is broke into following individual tests.
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [1, 8]
        option_split_batch = [BatchArgsOption.List, BatchArgsOption.Dict, BatchArgsOption.ListAndDict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_list_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [1]
        option_split_batch = [BatchArgsOption.List]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_dict_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [1]
        option_split_batch = [BatchArgsOption.Dict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_list_and_dict_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [1]
        option_split_batch = [BatchArgsOption.ListAndDict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_grad_accumulation_list_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [8]
        option_split_batch = [BatchArgsOption.List]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_grad_accumulation_dict_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [8]
        option_split_batch = [BatchArgsOption.Dict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

    def test_for_pretraining_full_precision_grad_accumulation_list_and_dict_input(self):
        option_fp16 = [False]
        option_allreduce_post_accumulation = [True]
        option_gradient_accumulation_steps = [8]
        option_split_batch = [BatchArgsOption.ListAndDict]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bert_for_pretraining(
            *config_and_inputs,
            option_fp16,
            option_allreduce_post_accumulation,
            option_gradient_accumulation_steps,
            option_split_batch)

if __name__ == "__main__":
    unittest.main()
