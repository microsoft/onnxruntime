# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from random import Random
from typing import Callable, Optional, Union

import transformers
from pydantic import Field, field_validator

from olive.common.config_utils import ConfigBase, validate_config, validate_object
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import StrEnumBase
from olive.data.component.dataset import ClassificationDataset
from olive.data.constants import IGNORE_INDEX


class TextGenStrategy(StrEnumBase):
    """Strategy for tokenizing a dataset."""

    LINE_BY_LINE = "line-by-line"  # each line is a sequence, in order of appearance
    LINE_BY_LINE_RANDOM = "line-by-line-random"  # each line is a sequence, in random order
    JOIN = "join"  # join all lines into a single sequence, split into non-overlapping sequences
    JOIN_RANDOM = "join-random"  # join all lines into a single sequence, split into random sequences
    JOIN_SLIDING_WINDOW = (  # join all lines into a single sequence, split into overlapping sequences
        "join-sliding-window"
    )


class TextGenParams(ConfigBase):
    """Common parameters for text generation tasks.

    Base dataclass for text generation tasks.
    """

    user_script: Optional[Union[str, Path]] = None  # user script use to define formatting functions
    script_dir: Optional[Union[str, Path]] = None  # directory with user script dependencies
    max_samples: Optional[int] = None  # max number of samples to use, None for all
    max_seq_len: int = 1024  # max length of sequence
    # TODO(jambayk): currently only support padding to max length since we preprocess all data at once
    # might have to expose collator for dataloader to support dynamic padding of batches
    # if false, cannot guarantee all sequences are same length. data loader will have to handle this during collation
    pad_to_max_len: bool = True  # pad sequences to max_len, ignored for JOIN corpus strategy
    padding_side: str = "right"  # pad to the right or left
    drop_short_sequences: bool = False  # drop sequences shorter than max_len. Mutually exclusive with pad_to_max_len
    use_attention_mask: bool = True  # add attention mask to each example
    # either use chat template or text
    # Chat template: template is applied to the chat messages to generate the text
    # use default (True) or custom (str) chat template
    # chat template must handle its own bos and eos tokens
    chat_template: Optional[Union[bool, str]] = None
    message_col: str = "messages"  # column with chat messages
    # Text: text is extracted from the dataset
    add_special_tokens: bool = True  # add bos and eos tokens to each sequence
    # one of text_formatting_func, text_template, or text_cols must be provided
    # priority: text_formatting_func > text_template > text_cols
    # function that formats the text, must take in an example dict and return a string
    text_formatting_func: Optional[Union[str, Callable]] = None  # name of formatting function for text
    # a python f-string template for the text with {column_name} as placeholders
    text_template: Optional[str] = None
    # list of text columns, columns are concatenated together using a space
    text_cols: Union[str, list[str]] = Field(default=["text"])
    # in JOIN strategies, the rows of text_cols are concatenated together
    strategy: TextGenStrategy = Field(default=TextGenStrategy.JOIN)
    stride: Optional[int] = None  # required when strategy is JOIN_SLIDING_WINDOW
    # text to join the rows of input columns when strategy is JOIN
    # add_special_tokens: "{bos_token} {text_col1} {eos_token} {joiner} {bos_token} {text_col2} {eos_token}..."
    # no add_special_tokens: "{text_col1} {joiner} {text_col2}..."
    # if None, joined with a space
    joiner: str = "\n\n"
    processing_batch_size: int = 1024  # number of examples to process at a time
    random_seed: Optional[int] = None  # random seed for LINE_BY_LINE_RANDOM and JOIN_RANDOM
    random_retries: int = (
        10  # number of resamples to try before giving up when a sample is too short for RANDOM strategies
    )

    @field_validator("padding_side", mode="before")
    @classmethod
    def _check_padding_side(cls, v):
        if v not in ["left", "right"]:
            raise ValueError("padding_side must be either left or right")
        return v

    @field_validator("drop_short_sequences", mode="before")
    @classmethod
    def _check_padding(cls, v, info):
        if "pad_to_max_len" not in info.data:
            raise ValueError("Invalid pad_to_max_len")
        if v and info.data["pad_to_max_len"]:
            raise ValueError("pad_to_max_len and drop_short_sequences cannot both be True")
        return v

    @field_validator("text_formatting_func", mode="before")
    @classmethod
    def _check_text_formatting_func(cls, v, info):
        # Create a simple object with name attribute for validate_object
        class FieldInfo:
            def __init__(self, name):
                self.name = name

        return validate_object(v, info.data, FieldInfo(info.field_name))

    @field_validator("text_cols", mode="before")
    @classmethod
    def _check_text_cols(cls, v, info):
        if "chat_template" not in info.data:
            raise ValueError("Invalid chat_template")

        if isinstance(v, str):
            v = [v]

        if info.data["chat_template"]:
            # chat template is used, so text_cols is not relevant
            return v

        alternatives = ["text_formatting_func", "text_template"]
        for alternate in alternatives:
            # for good validation error, check that all alternates are in values
            if alternate not in info.data:
                raise ValueError(f"Invalid {alternate}")

        if not (v or any(info.data[option] for option in alternatives)):
            # check that at least one alternate is specified
            raise ValueError(f"One of text_cols, {', '.join(alternatives)} must be specified")

        return v

    @field_validator("stride", mode="before")
    @classmethod
    def _check_stride(cls, v, info):
        if "strategy" not in info.data:
            raise ValueError("Invalid strategy")
        if info.data["strategy"] == TextGenStrategy.JOIN_SLIDING_WINDOW and v is None:
            raise ValueError("stride must be specified when strategy is JOIN_SLIDING_WINDOW")
        return v

    @field_validator("strategy", mode="before")
    @classmethod
    def _check_max_samples(cls, v, info):
        if "max_samples" not in info.data:
            raise ValueError("Invalid max_samples")
        if "random" in v and info.data["max_samples"] is None:
            raise ValueError("max_samples must be specified when strategy is random")
        return v

    @field_validator("strategy", mode="before")
    @classmethod
    def _check_use_attention_mask(cls, v, info):
        if "use_attention_mask" not in info.data:
            raise ValueError("Invalid use_attention_mask")
        if "pad_to_max_len" not in info.data:
            raise ValueError("Invalid pad_to_max_len")
        use_attention_mask = info.data["use_attention_mask"]
        pad_to_max_len = info.data["pad_to_max_len"]
        if "join" in v:
            # both True and False are valid since attention_mask is all 1s
            return v
        if not use_attention_mask and pad_to_max_len:
            raise ValueError(
                "pad_to_max_len is True but use_attention_mask is False. Attention mask is required for padding!"
            )
        return v

    @field_validator("random_seed", mode="before")
    @classmethod
    def _check_random(cls, v, info):
        if "strategy" not in info.data:
            raise ValueError("Invalid strategy")
        if "random" in info.data["strategy"] and v is None:
            raise ValueError("random_seed must be specified when strategy is random")
        return v

    def get_user_module_loader(self):
        """Get user module loader."""
        return UserModuleLoader(self.user_script, self.script_dir)


def text_gen_pre_process(dataset, tokenizer, all_kwargs):
    """Pre-process data for text generation task.

    The input dataset is expected to have one or more text columns.
    Depending on the strategy, the sequences are either joined together or processed individually.
    """
    import torch
    from datasets import Dataset as HFDataset

    args = validate_config(all_kwargs, TextGenParams, warn_unused_keys=True)

    # set tokenizer padding side
    tokenizer.padding_side = args.padding_side

    if isinstance(args.text_formatting_func, str):
        # load text_formatting_func
        args.text_formatting_func = args.get_user_module_loader().load_object(args.text_formatting_func)
    # get text from dataset
    dataset = dataset.map(
        lambda x: {
            "text": get_text(
                x,
                args.chat_template,
                args.message_col,
                args.text_formatting_func,
                args.text_template,
                args.text_cols,
                args.add_special_tokens,
                tokenizer,
            )
        }
    )
    text_list = [text for text in dataset["text"] if text]  # remove empty strings
    total_examples = len(text_list)  # total number of examples

    tokenized_inputs = {"input_ids": [], "labels": [], "attention_mask": []}
    if "join" in args.strategy:
        joiner_tokens = tokenizer.encode(args.joiner, add_special_tokens=False) if args.joiner else []

        if args.strategy != TextGenStrategy.JOIN_RANDOM:
            # no randomization, just use contiguous blocks of tokens
            if args.strategy == TextGenStrategy.JOIN_SLIDING_WINDOW:
                # we use the stride as both the step between sequences and the context size
                step, context = args.stride, args.max_seq_len - args.stride
            else:
                # JOIN strategy
                # text is split into non-overlapping sequences and there is no context
                step, context = args.max_seq_len, None

            example_idx = 0  # index of the first example in the current batch
            num_samples = 0  # samples processed so far
            overflow = []  # tokens overflowed from the previous batch of examples
            # we will process in batches to make tokenization faster
            # better than joining all text together and tokenizing all at once
            while True:
                if args.max_samples is not None and num_samples >= args.max_samples:
                    # we have reached max_samples
                    break
                if example_idx >= total_examples:
                    # we have reached the end of the text_list
                    break

                examples_to_get = min(args.processing_batch_size, total_examples - example_idx)
                # batch tokenize
                batched_input_ids = tokenizer(
                    text_list[example_idx : example_idx + examples_to_get],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                # join all the input_ids together with joiner_tokens
                joined_input_ids = overflow
                for input_ids in batched_input_ids:
                    joined_input_ids += input_ids + joiner_tokens

                end_loc = 0  # position of unused token in joined_input_ids
                # '- args.max_seq_len ' is used to make sure we don't get a sequence that is too short
                for begin_loc in range(0, len(joined_input_ids) - args.max_seq_len, step):
                    # end_loc is the beginning of the next sequence
                    end_loc = begin_loc + args.max_seq_len
                    # get the input sequence
                    input_ids = torch.tensor(joined_input_ids[begin_loc:end_loc])
                    append_text_gen_input_ids(tokenized_inputs, input_ids, torch.ones_like(input_ids), context=context)
                    num_samples += 1
                    if args.max_samples is not None and num_samples >= args.max_samples:
                        # we have reached max_samples
                        break
                # update counters
                example_idx += examples_to_get
                overflow = joined_input_ids[end_loc:]
        else:
            # randomization, sample random blocks of tokens
            rng = Random(args.random_seed)
            # cache to store tokenized examples
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                # will try to sample sequences random_retries times before giving up
                while resamples < args.random_retries:
                    # sample a beginning example
                    # randint is inclusive, so we need to subtract 1
                    begin_example_idx = rng.randint(0, total_examples - 1)
                    joined_input_ids = []
                    # loop through the examples until we have enough tokens
                    for i in range(begin_example_idx, total_examples):
                        # get the input_ids
                        if i not in cache:
                            cache[i] = tokenizer(
                                text_list[i],
                                add_special_tokens=False,
                                truncation=False,
                            )["input_ids"]
                        joined_input_ids += cache[i] + joiner_tokens
                        # stop if we have enough tokens
                        if len(joined_input_ids) >= args.max_seq_len:
                            break
                    # add to samples if we have enough tokens
                    if len(joined_input_ids) >= args.max_seq_len:
                        # found a good example
                        input_ids = torch.tensor(joined_input_ids[: args.max_seq_len])
                        append_text_gen_input_ids(tokenized_inputs, input_ids, torch.ones_like(input_ids))
                        break
                    resamples += 1
    else:
        # each line is a sequence
        if args.strategy == TextGenStrategy.LINE_BY_LINE:
            # batched tokenization might be faster so lets tokenize all the text at once
            if not args.max_samples:
                for native_input_ids, native_attention_mask in batch_tokenize_text(text_list, tokenizer, args):
                    append_text_gen_input_ids(
                        tokenized_inputs, torch.tensor(native_input_ids), torch.tensor(native_attention_mask)
                    )
            else:
                example_idx = 0  # index of the first example in the current batch
                num_samples = 0
                while True:
                    if num_samples >= args.max_samples or example_idx >= total_examples:
                        # we have reached max_samples or the end of the text_list
                        break
                    # get as many examples as possible without going over max_samples
                    examples_to_get = min(args.max_samples - num_samples, total_examples - example_idx)
                    # batch tokenize
                    tokenized_texts = batch_tokenize_text(
                        text_list[example_idx : example_idx + examples_to_get],
                        tokenizer,
                        args,
                    )
                    for native_input_ids, native_attention_mask in tokenized_texts:
                        append_text_gen_input_ids(
                            tokenized_inputs, torch.tensor(native_input_ids), torch.tensor(native_attention_mask)
                        )
                        num_samples += 1
                    # update counters
                    example_idx += examples_to_get
        else:
            # randomization, sample random lines
            rng = Random(args.random_seed)
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                encodings = None
                while resamples < args.random_retries:
                    # sample a random line
                    # randint is inclusive, so we need to subtract 1
                    i = rng.randint(0, len(text_list) - 1)
                    if i not in cache:
                        encodings = tokenizer(
                            text_list[i],
                            max_length=args.max_seq_len,
                            truncation=True,
                            padding="max_length" if args.pad_to_max_len else False,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        cache[i] = encodings
                    else:
                        encodings = cache[i]
                    if not args.drop_short_sequences or encodings.input_ids.shape[1] >= args.max_seq_len:
                        # found a good sample
                        break
                    resamples += 1
                if not encodings:
                    # could not find a good sample after resampling
                    continue
                append_text_gen_input_ids(tokenized_inputs, encodings.input_ids[0], encodings.attention_mask[0])

    if not args.use_attention_mask:
        # remove attention_mask
        tokenized_inputs.pop("attention_mask")

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    # return ClassificationDataset
    return ClassificationDataset(hf_dataset, "labels", max_samples=args.max_samples)


def get_text(
    example: dict[str, str],
    chat_template: Optional[Union[bool, str]] = None,
    message_col: str = "messages",
    formatting_func: Optional[Callable] = None,
    template: Optional[str] = None,
    cols: Optional[list[str]] = None,
    add_special_tokens: bool = False,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
):
    """Get text from example using formatting_func, template, or cols."""
    if chat_template:
        # do we need to add generation prompt?
        return tokenizer.apply_chat_template(
            example[message_col],
            tokenize=False,
            chat_template=chat_template if isinstance(chat_template, str) else None,
        )

    if formatting_func:
        text = formatting_func(example)
    elif template:
        text = template.format(**example)
    elif cols:
        text = " ".join([example[col] for col in cols])
    else:
        raise ValueError("None of formatting_func, template, or cols is specified")
    if add_special_tokens:
        # add bos and eos tokens
        text = f"{tokenizer.bos_token} {text} {tokenizer.eos_token}"
    return text


def batch_tokenize_text(text_list, tokenizer, args):
    """Batch tokenize text."""
    batched_encodings = tokenizer(
        text_list,
        max_length=args.max_seq_len,
        truncation=True,
        padding="max_length" if args.pad_to_max_len else False,
        add_special_tokens=False,
    )
    batched_encodings = zip(batched_encodings.input_ids, batched_encodings.attention_mask)
    if args.drop_short_sequences:
        batched_encodings = filter(lambda encoding: len(encoding[0]) >= args.max_seq_len, batched_encodings)
    return batched_encodings


def append_text_gen_input_ids(
    tokenized_inputs, input_ids, attention_mask, context: Optional[int] = None, ignore_index=IGNORE_INDEX
):
    """Convert input_ids to inputs dict and append to tokenized_inputs."""
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # create labels
    # target is not shifted by 1 since causal lm models shifts internally when computing loss
    inputs["labels"] = labels = input_ids.clone()
    # set context to ignore_index
    if context is not None:
        labels[:context] = ignore_index
    # set padding to ignore_index
    labels[attention_mask != 1] = ignore_index

    # add to list
    for k, v in inputs.items():
        tokenized_inputs[k].append(v)
