# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

try:
    from lm_eval.models.utils_hf import pad_and_concat  # pylint: disable=ungrouped-imports
except ImportError:
    from lm_eval.models.utils import pad_and_concat

from olive.common.onnx_io import get_io_config, get_io_dtypes, get_kv_info
from olive.common.utils import cleanup_memory

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

logger = logging.getLogger(__name__)

LogLikelihoodInputs = tuple[tuple[str, str], list[int], list[int]]


class LMEvalOnnxBase(TemplateLM):
    """Base class for ONNX model evaluation."""

    @abstractmethod
    def prepare(self, requests: list[LogLikelihoodInputs]):
        pass

    @abstractmethod
    def model_call(self, input_ids: torch.Tensor, cont_len: int = 0) -> torch.Tensor:
        pass

    @abstractmethod
    def complete(self):
        pass

    def _loglikelihood_tokens(self, requests: list[LogLikelihoodInputs], **kwargs) -> list[tuple[float, bool]]:
        self.prepare(requests)

        def _collate(req: LogLikelihoodInputs):
            """Return the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: LogLikelihoodInputs):
            """Return the key to group and lookup one-token continuations."""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts",
            group_fn=_lookup_one_token_cont,
        )

        res = []

        logger.info("Calculating loglikelihood for %d requests", len(re_ord))
        pbar = tqdm(total=len(re_ord), desc="Running loglikelihood requests")
        for chunk in re_ord.get_batched(n=self.batch_size):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length + 1:
                    logger.warning(
                        "Combined length of context %d and continuation %d exceeds model's maximum length (%d)."
                        " Truncating %d tokens from the left.",
                        len(context_enc),
                        len(continuation_enc),
                        self.max_length,
                        total_length - self.max_length + 1,
                    )
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                )
                (inplen,) = inp.shape

                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")  # [batch, padding_len_inp]
            max_cont_len = max(len(c) for c in cont_toks_list)

            multi_logits = self.model_call(batched_inps, max_cont_len)  # [batch, padding_length (inp or cont), vocab]

            # ruff: noqa: PLW2901
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = logits[ctx_len - contlen : ctx_len]
                logits = logits.unsqueeze(0)  # [1, seq, vocab]
                logits = F.log_softmax(logits, dim=-1)

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # ruff: noqa: B020
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long, device=greedy_tokens.device).unsqueeze(
                        0
                    )  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()
        self.complete()

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")


@register_model("ort")
class LMEvalORTEvaluator(LMEvalOnnxBase):
    """Evaluate a model using ONNX Runtime and IOBinding."""

    _DEFAULT_MAX_LENGTH = 2048
    _TOKENIZER_INFINITY = 1000000000000000019884624838656

    def __init__(
        self,
        model_path: str,
        batch_size: int | str = 1,
        max_length: int | None = None,
        ep: str | None = None,
        ep_options: str | None = None,
        add_bos_token: bool | None = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.prefill = Prefill(model_path, ep, ep_options)
        self.config = AutoConfig.from_pretrained(Path(model_path).parent)
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_path).parent)

        self.add_bos_token = add_bos_token

        # consider adding auto batch sizes
        self.batch_size = int(batch_size)
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == self._TOKENIZER_INFINITY:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        return self.tokenizer.encode(string, **special_tokens_kwargs)

    def prepare(self, requests: list[LogLikelihoodInputs]):
        max_length = -1
        for _, context_enc, continuation_enc in requests:
            max_length = max(max_length, len(context_enc) + len(continuation_enc) - 1)
        max_length = min(max_length, self.max_length)
        self.prefill.initialize_buffers(self.batch_size, max_length)

    def model_call(self, input_ids: torch.Tensor, cont_len: int = 0) -> torch.Tensor:
        return self.prefill.run(input_ids)

    def complete(self):
        self.prefill.reset_buffers()


class Prefill:
    """Class to run a single inference pass with an ONNX LLM model using ONNXRuntime + IOBinding."""

    def __init__(self, model_path: str, ep: str | None = None, ep_options: dict | None = None):
        """Initialize the Prefill class.

        :param model_path: Path to the ONNX model file. model_path.parent must have the transformers tokenizer and config files
        :param ep: Execution provider to use for inference. If None, defaults to CPU.
        :param ep_options: Options for the execution provider.
        """
        self.model_path = model_path

        if ep == "CUDAExecutionProvider" and not torch.cuda.is_available():
            raise RuntimeError("CUDAExecutionProvider requires torch.cuda to be available")
        self.ep = ep
        self.ep_options = ep_options
        # device to do IOBinding on
        self.device = "cuda" if ep == "CUDAExecutionProvider" else "cpu"

        # model's io info
        self.io_config = get_io_config(self.model_path)
        self.io_dtypes = get_io_dtypes(self.io_config)
        self.vocab = dict(zip(self.io_config["output_names"], self.io_config["output_shapes"]))["logits"][-1]
        self.kv_info = get_kv_info(self.io_config)
        if self.kv_info is None:
            raise ValueError("Invalid io_config: kv_info not found")

        self._session = None
        self._batch_size = None
        self._buffers = None

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a single inference pass with the ONNX model.

        :param input_ids: Input tensor containing token IDs with shape [batch_size, seqlen]
        :return: Output tensor containing logits with shape [batch_size, seqlen, vocab_size]
        """
        io_binding = self.session.io_binding()

        batch_size, seqlen = input_ids.shape
        if batch_size > self._batch_size:
            raise ValueError(f"Invalid batch size: {batch_size}, max batch size: {self._batch_size}")

        # bind inputs
        inputs_to_bind = {
            "input_ids": (
                input_ids.to(device=self.device, dtype=getattr(torch, self.io_dtypes["input_ids"])).contiguous(),
                self.io_dtypes["input_ids"],
                (batch_size, seqlen),
            ),
        }
        if "past_seq_len" in self._buffers["inputs"]:
            self._buffers["inputs"]["past_seq_len"][:] = seqlen - 1
            self._buffers["inputs"]["total_seq_len"][:] = seqlen
        for name, shape in [
            ("attention_mask", (batch_size, seqlen)),
            ("past_seq_len", (batch_size, 1)),
            ("total_seq_len", (1,)),
        ]:
            if name not in self._buffers["inputs"]:
                continue
            # no need to slice extra elements: attention_mask is all 1s, other inputs have fixed shapes
            inputs_to_bind[name] = (self._buffers["inputs"][name], self.io_dtypes[name], shape)
        if "position_ids" in self._buffers["inputs"]:
            # need to reallocate since the position_ids tensor may be sliced
            inputs_to_bind["position_ids"] = (
                self._buffers["inputs"]["position_ids"][:batch_size, :seqlen].contiguous(),
                self.io_dtypes["position_ids"],
                (batch_size, seqlen),
            )
        for name in self._buffers["kv_inputs"]:
            inputs_to_bind[name] = (
                self._buffers["kv_inputs"][name],
                self.kv_info["dtype"],
                (batch_size, self.kv_info["num_kv_heads"], 0, self.kv_info["head_size"]),
            )
        for name, (buffer, dtype, shape) in inputs_to_bind.items():
            io_binding.bind_input(
                name,
                device_type=self.device,
                device_id=0,
                element_type=dtype,
                shape=shape,
                buffer_ptr=buffer.data_ptr(),
            )

        # bind outputs
        outputs_to_bind = {
            # provide full buffer, will slice batch_size * seqlen * self.vocab elements after run
            "logits": (self._buffers["outputs"]["logits"], self.io_dtypes["logits"], (batch_size, seqlen, self.vocab)),
        }
        for name in self._buffers["kv_outputs"]:
            outputs_to_bind[name] = (
                self._buffers["kv_outputs"][name],
                self.kv_info["dtype"],
                (batch_size, self.kv_info["num_kv_heads"], seqlen, self.kv_info["head_size"]),
            )
        for name, (buffer, dtype, shape) in outputs_to_bind.items():
            io_binding.bind_output(
                name,
                device_type=self.device,
                device_id=0,
                element_type=dtype,
                shape=shape,
                buffer_ptr=buffer.data_ptr(),
            )

        io_binding.synchronize_inputs()
        self.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        return self._buffers["outputs"]["logits"][: batch_size * seqlen * self.vocab].view(
            batch_size, seqlen, self.vocab
        )

    @property
    def session(self):
        # TODO(jambayk): use olive.common.ort_inference.get_ort_inference_session instead
        # need to sort out where the device and ep comes from
        from onnxruntime import InferenceSession

        if self._session is not None:
            return self._session

        self._session = InferenceSession(
            self.model_path,
            providers=[self.ep] if self.ep else None,
            provider_options=[self.ep_options] if self.ep_options else None,
        )
        return self._session

    def reset_buffers(self):
        """Reset the input and output buffers."""
        self._buffers = None
        self._batch_size = None
        cleanup_memory()

    def initialize_buffers(self, batch_size: int, max_length: int):
        """Initialize input and output buffers to use for IOBinding.

        :param batch_size: The maximum batch size for the input tensors.
        :param max_length: The maximum sequence length for the input tensors.
        """
        self.reset_buffers()

        # inputs other than kv cache
        inputs = {
            "attention_mask": torch.ones(
                batch_size * max_length, dtype=getattr(torch, self.io_dtypes["attention_mask"]), device=self.device
            )
        }
        if self.io_dtypes.get("position_ids") is not None:
            inputs["position_ids"] = (
                torch.arange(max_length, dtype=getattr(torch, self.io_dtypes["position_ids"]), device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        if self.io_dtypes.get("past_seq_len") is not None:
            inputs["past_seq_len"] = (
                torch.tensor(max_length - 1, dtype=getattr(torch, self.io_dtypes["past_seq_len"]), device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            inputs["total_seq_len"] = torch.tensor(
                max_length, dtype=getattr(torch, self.io_dtypes["total_seq_len"]), device=self.device
            ).unsqueeze(0)

        # outputs other than kv cache
        outputs = {
            "logits": torch.zeros(
                batch_size * max_length * self.vocab, dtype=getattr(torch, self.io_dtypes["logits"]), device=self.device
            )
        }

        # kv cache inputs
        kv_inputs = {
            name: torch.zeros(0, dtype=getattr(torch, self.kv_info["dtype"]), device=self.device)
            for name in self.kv_info["past_names"]
        }

        # kv cache outputs
        kv_outputs = {
            name: torch.zeros(
                batch_size * self.kv_info["num_kv_heads"] * max_length * self.kv_info["head_size"],
                dtype=getattr(torch, self.kv_info["dtype"]),
                device=self.device,
            )
            for name in self.kv_info["present_to_past"]
        }

        self._buffers = {"inputs": inputs, "outputs": outputs, "kv_inputs": kv_inputs, "kv_outputs": kv_outputs}
        self._batch_size = batch_size


@register_model("ortgenai")
class LMEvalORTGenAIEvaluator(LMEvalOnnxBase):
    """Evaluate a model using ONNX Runtime GenAI."""

    def __init__(
        self,
        pretrained: str,
        batch_size: int | str = 1,
        max_length: int | None = None,
        ep: str = "follow_config",
        ep_options: dict | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize the evaluator.

        :param pretrained: The path to the model directory
        :param batch_size: The batch size to use for evaluation
        :param max_length: The maximum sequence length for evaluation
        :param ep: The execution provider to use. "follow_config" will use the provider specified in the genai_config file
        :param ep_options: The options to use for the execution provider. Only applicable if ep is not "follow_config"
        :param device: The device to run log likelihood calculations on
        """
        if og is None:
            raise ImportError("onnxruntime-genai is not installed.")

        super().__init__()

        self.config = og.Config(pretrained)
        if ep != "follow_config":
            ep = ep.lower().replace("executionprovider", "")
            self.config.clear_providers()
            if ep != "cpu":
                self.config.append_provider(ep)
            for key, value in (ep_options or {}).items():
                self.config.set_provider_option(ep, key, value)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)

        # consider adding auto batch sizes
        self.batch_size = int(batch_size)

        with (Path(pretrained) / "genai_config.json").open() as f:
            genai_config = json.load(f)

            if max_length:
                self.max_length = max_length
            else:
                self.max_length = genai_config["search"]["max_length"]
            self._eot_token_id = genai_config["model"]["eos_token_id"]
        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(max_length=self.max_length, past_present_share_buffer=False)

        self.device = device
        self._returns_full_logits = self._detect_full_logits()

    def _detect_full_logits(self) -> bool:
        """Check if the model returns logits for all input positions or only the last."""
        try:
            dummy_len = 3
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=self.max_length, past_present_share_buffer=False, batch_size=1)
            generator = og.Generator(self.model, params)
            dummy_ids = [[self._eot_token_id] * dummy_len]
            generator.append_tokens(dummy_ids)
            logits = generator.get_output("logits")
            return logits.shape[1] == dummy_len
        except Exception:
            return False

    @property
    def eot_token_id(self):
        return self._eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string using the model's tokenizer and return a list of token IDs."""
        return self.tokenizer.encode(string).tolist()

    def prepare(self, requests: list[LogLikelihoodInputs]):
        pass

    def model_call(self, input_ids: torch.Tensor, cont_len: int = 0) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        self.params.set_search_options(batch_size=batch_size)
        generator = og.Generator(self.model, self.params)

        if self._returns_full_logits:
            generator.append_tokens(input_ids.tolist())
            return torch.from_numpy(generator.get_output("logits")).to(self.device)

        # Model only returns logits for the last appended position.
        if batch_size > 1 and cont_len > 1:
            raise ValueError(
                "batch_size > 1 is not supported when the model returns single-position logits"
                " and continuation length > 1. Right-padding misaligns continuation positions across"
                " batch elements. Use batch_size=1 instead."
            )

        # Bulk-append context tokens, then step through the last cont_len tokens
        # one at a time to collect only the logits we actually need.
        n_logits = max(cont_len, 1)
        prefix_len = seq_len - n_logits
        generator.append_tokens(input_ids[:, : prefix_len + 1].tolist())
        all_logits = [torch.from_numpy(generator.get_output("logits")).to(self.device)]
        for i in range(prefix_len + 1, seq_len):
            generator.append_tokens(input_ids[:, i : i + 1].tolist())
            all_logits.append(torch.from_numpy(generator.get_output("logits")).to(self.device))

        # No need to pad to [batch, seq_len, vocab]. The slicing in _loglikelihood_tokens computes
        # ctx_len = inplen + (logits.shape[0] - padding_len_inp), which adjusts for the shorter
        # seq dimension so the continuation slice still lands on the correct positions.
        return torch.cat(all_logits, dim=1)  # [batch, n_logits, vocab]

    def complete(self):
        pass
