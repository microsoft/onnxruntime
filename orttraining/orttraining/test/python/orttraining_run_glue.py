# adapted from run_glue.py of huggingface transformers

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import unittest
import numpy as np
from numpy.testing import assert_allclose

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription

try:
    from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size,\
        get_mpi_context_world_rank, get_mpi_context_world_size
    has_get_mpi_context_internal_api = True
except ImportError:
    has_get_mpi_context_internal_api = False
    pass


from orttraining_transformer_trainer import ORTTransformerTrainer

import torch

logger = logging.getLogger(__name__)

def verify_old_and_new_api_are_equal(results_per_api):
    new_api_results = results_per_api[True]
    old_api_results = results_per_api[False]
    for key in new_api_results.keys():
        assert_allclose(new_api_results[key], old_api_results[key])

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

class ORTGlueTest(unittest.TestCase):

    def setUp(self):
        # configurations not to be changed accoss tests
        self.max_seq_length = 128
        self.train_batch_size = 8
        self.learning_rate = 2e-5
        self.num_train_epochs = 3.0
        self.local_rank = -1
        self.world_size = 1
        self.overwrite_output_dir = True
        self.gradient_accumulation_steps = 1
        self.data_dir = "/bert_data/hf_data/glue_data/"
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glue_test_output/")
        self.cache_dir = '/tmp/glue/'
        self.logging_steps = 10

    def test_roberta_with_mrpc(self):
        expected_acc = 0.85
        expected_f1 = 0.88
        expected_loss = 0.35
        results = self.run_glue(model_name="roberta-base", task_name="MRPC", fp16=False)

        assert(results['acc'] >= expected_acc)
        assert(results['f1'] >= expected_f1)
        assert(results['loss'] <= expected_loss)

    def test_roberta_fp16_with_mrpc(self):
        expected_acc = 0.87
        expected_f1 = 0.90
        expected_loss = 0.33

        results = self.run_glue(model_name="roberta-base", task_name="MRPC", fp16=True)

        assert(results['acc'] >= expected_acc)
        assert(results['f1'] >= expected_f1)
        assert(results['loss'] <= expected_loss)

    def test_bert_with_mrpc(self):
        if self.local_rank == -1:
            expected_acc = 0.83
            expected_f1 = 0.88
            expected_loss = 0.44
        elif self.local_rank == 0:
            expected_acc = 0.81
            expected_f1 = 0.86
            expected_loss = 0.44

        results = self.run_glue(model_name="bert-base-cased", task_name="MRPC", fp16=False)

        if self.local_rank in [-1, 0]:
            assert(results['acc'] >= expected_acc)
            assert(results['f1'] >= expected_f1)
            assert(results['loss'] <= expected_loss)

    def test_bert_fp16_with_mrpc(self):
        expected_acc = 0.84
        expected_f1 = 0.88
        expected_loss = 0.46

        results = self.run_glue(model_name="bert-base-cased", task_name="MRPC", fp16=True)

        assert(results['acc'] >= expected_acc)
        assert(results['f1'] >= expected_f1)
        assert(results['loss'] <= expected_loss)

    def model_to_desc(self, model_name, model):
        if model_name.startswith('bert') or model_name.startswith('xlnet'):
            model_desc = {
                'inputs': [
                    ('input_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
                    ('token_type_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('labels', ['batch', ],)],
                'outputs': [('loss', [], True),
                            ('logits', ['batch', 2])]}
        elif model_name.startswith('roberta'):
            model_desc = {
                'inputs': [
                    ('input_ids', ['batch', 'max_seq_len_in_batch'],),
                    ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
                    ('labels', ['batch', ],)],
                'outputs': [('loss', [], True),
                            ('logits', ['batch', 2])]}
        else:
            raise RuntimeError("unsupported base model name {}.".format(model_name))

        return model_desc

    def run_glue(self, model_name, task_name, fp16):
        model_args = ModelArguments(model_name_or_path=model_name, cache_dir=self.cache_dir)
        data_args = GlueDataTrainingArguments(
            task_name=task_name, data_dir=os.path.join(self.data_dir, task_name),
            max_seq_length=self.max_seq_length)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, task_name), do_train=True, do_eval=True,
            per_gpu_train_batch_size=self.train_batch_size,
            learning_rate=self.learning_rate, num_train_epochs=self.num_train_epochs,
            local_rank=self.local_rank,
            overwrite_output_dir=self.overwrite_output_dir, gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=fp16, logging_steps=self.logging_steps)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        set_seed(training_args.seed)
        onnxruntime.set_seed(training_args.seed)

        try:
            num_labels = glue_tasks_num_labels[data_args.task_name]
            output_mode = glue_output_modes[data_args.task_name]
        except KeyError:
            raise ValueError("Task not found: %s" % (data_args.task_name))

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        train_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer)
            if training_args.do_train
            else None
        )

        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
            if training_args.do_eval
            else None
        )

        def compute_metrics(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        model_desc = self.model_to_desc(model_name, model)
        # Initialize the ORTTrainer within ORTTransformerTrainer
        trainer = ORTTransformerTrainer(
            model=model,
            model_desc=model_desc,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            world_size=self.world_size,
        )

        # Training
        if training_args.do_train:
            trainer.train()
            trainer.save_model()

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()

            logger.info("***** Eval results {} *****".format(data_args.task_name))
            for key, value in result.items():
               logger.info("  %s = %s", key, value)

            results.update(result)

        return results

if __name__ == "__main__":
    if has_get_mpi_context_internal_api:
        local_rank = get_mpi_context_local_rank()
        world_size = get_mpi_context_world_size()
    else:
        local_rank = -1
        world_size = 1

    if world_size > 1:
        # mpi launch
        logger.warning("mpirun launch, local_rank / world_size: %s : % s", local_rank, world_size)

        # TrainingArguments._setup_devices will call torch.distributed.init_process_group(backend="nccl")
        # pytorch expects following environment settings (which would be set if launched with torch.distributed.launch).

        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        from onnxruntime.capi._pybind_state import set_cuda_device_id
        set_cuda_device_id(local_rank)

        test = ORTGlueTest()
        test.setUp()
        test.local_rank = local_rank
        test.world_size = world_size
        test.test_bert_with_mrpc()
    else:
        unittest.main()
