# adapted from run_multiple_choice.py of huggingface transformers
# https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/run_multiple_choice.py

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
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription

from orttraining_transformer_trainer import ORTTransformerTrainer

import torch

from utils_multiple_choice import MultipleChoiceDataset, Split, SwagProcessor

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on."})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

class ORTMultipleChoiceTest(unittest.TestCase):

    def setUp(self):
        # configurations not to be changed accoss tests
        self.max_seq_length = 80
        self.train_batch_size = 2
        self.eval_batch_size = 2
        self.learning_rate = 2e-5
        self.num_train_epochs = 3.0
        self.local_rank = -1
        self.overwrite_output_dir = True
        self.gradient_accumulation_steps = 8
        self.data_dir = "/bert_data/hf_data/swag/swagaf/data"
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "multiple_choice_test_output/")
        self.cache_dir = '/tmp/multiple_choice/'
        self.logging_steps = 10

    def test_bert_with_swag(self):
        expected_acc = 0.7883135059482156
        expected_loss = 0.6474186203172139

        results = self.run_multiple_choice(model_name="bert-base-cased", task_name="swag", fp16=False)
        assert_allclose(results['acc'], expected_acc)
        assert_allclose(results['loss'], expected_loss)

    def test_bert_fp16_with_swag(self):
        expected_acc = 0.7876137158852344
        expected_loss = 0.6469138865876788

        results = self.run_multiple_choice(model_name="bert-base-cased", task_name="swag", fp16=True)
        assert_allclose(results['acc'], expected_acc)
        assert_allclose(results['loss'], expected_loss)

    def run_multiple_choice(self, model_name, task_name, fp16):
        model_args = ModelArguments(model_name_or_path=model_name, cache_dir=self.cache_dir)
        data_args = DataTrainingArguments(task_name=task_name, data_dir=self.data_dir,
            max_seq_length=self.max_seq_length)

        training_args = TrainingArguments(output_dir=os.path.join(self.output_dir, task_name), do_train=True, do_eval=True,
            per_gpu_train_batch_size=self.train_batch_size,
            per_gpu_eval_batch_size=self.eval_batch_size,
            learning_rate=self.learning_rate, num_train_epochs=self.num_train_epochs,local_rank=self.local_rank,
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
            processor = SwagProcessor()
            label_list = processor.get_labels()
            num_labels = len(label_list)
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

        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        # Get datasets
        train_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                processor=processor,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                processor=processor,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            if training_args.do_eval
            else None
        )

        def compute_metrics(p: EvalPrediction) -> Dict:
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": simple_accuracy(preds, p.label_ids)}

        if model_name.startswith('bert'):
            model_desc = ModelDescription([
                IODescription('input_ids', [self.train_batch_size, num_labels, data_args.max_seq_length], torch.int64, num_classes=model.config.vocab_size),
                IODescription('attention_mask', [self.train_batch_size, num_labels, data_args.max_seq_length], torch.int64, num_classes=2),
                IODescription('token_type_ids', [self.train_batch_size, num_labels, data_args.max_seq_length], torch.int64, num_classes=2),
                IODescription('labels', [self.train_batch_size, num_labels], torch.int64, num_classes=num_labels)], [
                IODescription('loss', [], torch.float32),
                IODescription('reshaped_logits', [self.train_batch_size, num_labels], torch.float32)])
        else:
            model_desc = ModelDescription([
                IODescription('input_ids', ['batch', num_labels, 'max_seq_len_in_batch'], torch.int64, num_classes=model.config.vocab_size),
                IODescription('attention_mask', ['batch', num_labels, 'max_seq_len_in_batch'], torch.int64, num_classes=2),
                IODescription('labels', ['batch', num_labels], torch.int64, num_classes=num_labels)], [
                IODescription('loss', [], torch.float32),
                IODescription('reshaped_logits', ['batch', num_labels], torch.float32)])

        # Initialize the ORTTrainer within ORTTransformerTrainer
        trainer = ORTTransformerTrainer(
            model=model,
            model_desc=model_desc,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
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
    unittest.main()
