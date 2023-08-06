##########################
# Benchmark Whisper model
##########################

import argparse
import datetime
import gc
import logging
import os
import sys
import time

import librosa
import numpy as np
import psutil
import torch
import whisper
from onnxruntime_extensions import get_library_path
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.pipelines import pipeline as ort_pipeline
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import trange
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import pipeline as pt_pipeline

import onnxruntime as ort
from onnxruntime.transformers.benchmark_helper import measure_memory

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_helper import setup_logger  # noqa: E402

PRECISION = {
    "fp32": (torch.float32, np.float32),
    "fp16": (torch.float16, np.float16),
    "int8": (torch.int8, np.int8),
}

logger = logging.getLogger(__name__)


# features_type: "" for none, "pt" for PyTorch, or "np" for NumPy
# load_with: "ffmpeg" for whisper.load_audio, "librosa" for librosa.load, "basic" for using AudioDecoder in ONNX model
def get_audio(args, load_with="ffmpeg", features_type=""):
    if features_type != "":
        assert load_with == "ffmpeg"

    def load_via_ffmpeg():
        audio = whisper.load_audio(args.audio_path)
        audio = whisper.pad_or_trim(audio)
        return audio

    def load_via_librosa():
        audio = librosa.load(args.audio_path)[0]
        audio = np.expand_dims(audio[: 30 * args.sample_rate], axis=0)
        return audio

    def load_via_basic():
        with open(args.audio_path, "rb") as fobj:
            audio = np.asarray(list(fobj.read()), dtype=np.uint8)
            audio = np.array([audio] * args.batch_size)
        return audio

    load_audio = (
        load_via_ffmpeg if load_with == "ffmpeg" else load_via_librosa if load_with == "librosa" else load_via_basic
    )

    load_audio_fn = lambda dummy_input: load_audio()  # noqa: E731
    audio, metrics = time_fn(args, load_audio_fn, None)
    logger.info(f"Load audio: {metrics[1]} s")

    if features_type == "":
        return audio

    processor = AutoProcessor.from_pretrained(args.model_name)

    feature_extractor_fn = lambda audio: processor.feature_extractor(
        [audio] * args.batch_size, return_tensors=features_type
    ).input_features  # noqa: E731
    input_features, metrics = time_fn(args, feature_extractor_fn, audio)
    logger.info(f"Feature extraction: {metrics[1]} s")

    return input_features


def get_ort_inputs(args):
    if "encoder_model" in args.ort_model_path:
        # Encoder component of "Whisper export with optimum"
        ort_inputs = {
            "input_features": np.random.rand(args.batch_size, args.feature_size, args.encoder_seq_len).astype(
                np.float32
            ),
        }
        exclude_list = []
    elif "decoder_model" in args.ort_model_path:
        # Decoder component of "Whisper export with optimum"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, args.decoder_seq_len).astype(np.int64),
            "encoder_hidden_states": np.random.rand(
                args.batch_size, args.encoder_seq_len // 2, args.hidden_size
            ).astype(np.float32),
        }
        exclude_list = ["input_ids"]
    elif "decoder_with_past_model" in args.ort_model_path:
        # Decoder-with-past component of "Whisper export with optimum"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, 1).astype(np.int64),
        }
        for i in range(args.num_layers):
            past_kv = {
                f"past_key_values.{i}.decoder.key": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.decoder.value": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.encoder.key": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.encoder.value": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
            }
            ort_inputs.update(past_kv)
        exclude_list = ["input_ids"]
    elif "_encoder_decoder_init" in args.ort_model_path:
        # Encoder-decoder-init component of "Whisper custom export with beam search"
        ort_inputs = {
            "encoder_input_ids": np.random.rand(args.batch_size, args.feature_size, args.encoder_seq_len).astype(
                np.float32
            ),
            "decoder_input_ids": np.random.rand(args.batch_size, 1).astype(np.int32),
        }
        exclude_list = ["decoder_input_ids"]
    elif "_decoder" in args.ort_model_path:
        # Decoder-with-past component of "Whisper custom export with beam search"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, 1).astype(np.int32),
        }
        for i in range(args.num_layers):
            past_kv = {
                f"past_key_self_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_value_self_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_cross_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
                f"past_value_cross_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
            }
            ort_inputs.update(past_kv)
        exclude_list = ["input_ids"]
    elif "beamsearch" in args.ort_model_path:
        # Whisper custom export with beam search contrib op
        input_features = get_audio(args, load_with="ffmpeg", features_type="np")
        ort_inputs = {
            "input_features": input_features,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            "decoder_input_ids": np.array([args.decoder_input_ids], dtype=np.int32),
            "logits_processor": np.array([args.logits_processor], dtype=np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    elif "all" in args.ort_model_path or "large-v2" in args.ort_model_path:
        # Whisper end-to-end ONNX model
        audio = get_audio(args, load_with="basic") if args.audio_path != "" else None
        ort_inputs = {
            "audio_stream": audio,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            "decoder_input_ids": np.array([args.decoder_input_ids], dtype=np.int32),
            "logits_processor": np.array([args.logits_processor], dtype=np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    else:
        raise Exception("Unable to auto-detect inputs for provided model")

    return set_inputs(args, ort_inputs, exclude_list)


def get_hf_inputs(args, processor):
    if args.hf_api == "pipeline":
        # Only the audio is needed for inputs
        audio = get_audio(args, load_with="ffmpeg")
        hf_inputs = {"audio": audio}
        exclude_list = ["audio"]
    elif args.hf_api == "gen-and-dec":
        target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device
        input_features = get_audio(args, load_with="ffmpeg", features_type="pt")
        hf_inputs = {
            "inputs": input_features.to(target_device),
            "max_length": args.max_length,
            "min_length": args.min_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
            "use_cache": True,
        }
        exclude_list = [key for key in hf_inputs if key not in {"inputs"}]
    else:
        raise Exception("Could not calculate model inputs")

    return set_inputs(args, hf_inputs, exclude_list)


def set_inputs(args, input_dict, exclude_list):
    # Cast certain inputs to another dtype
    precision_dest = "fp32" if args.precision == "int8" else args.precision

    for k, v in input_dict.items():
        if k in exclude_list:
            continue

        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(PRECISION[precision_dest][0])
        elif isinstance(v, np.ndarray):
            input_dict[k] = v.astype(PRECISION[precision_dest][1])

    return input_dict


def get_vars(args):
    inputs, processor, model, pipeline = None, None, None, None
    if args.benchmark_type in {"HF + PT", "HF + PT2"}:
        processor, model, pipeline = get_hf_pt(args)
    elif args.benchmark_type == "HF + ORT":
        processor, model, pipeline = get_hf_ort(args)
    elif args.benchmark_type == "ORT":
        model = get_ort_model(args)
    else:
        raise Exception("Invalid benchmark type provided")

    # Get inputs
    if args.benchmark_type == "ORT":
        inputs = get_ort_inputs(args)
        if (
            "audio_stream" not in set(map(lambda model_input: model_input.name, model.get_inputs()))
            and "audio_stream" in inputs
        ):
            # Remove when 'audio' input is not in model
            del inputs["audio_stream"]
    else:
        inputs = get_hf_inputs(args, processor)
        if args.hf_api == "pipeline":
            inputs = inputs["audio"]

    return inputs, processor, model, pipeline


def get_hf_pt(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    torch_dtype = PRECISION[args.precision][0] if args.precision != "int8" else PRECISION["fp32"][0]
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device

    if args.hf_pt_model_path == "":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
        ).to(target_device)
    else:
        assert os.path.exists(args.hf_pt_model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.hf_pt_model_path,
            torch_dtype=torch_dtype,
        ).to(target_device)

    if "PT2" in args.benchmark_type:
        model = torch.compile(model)

    pipeline = pt_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=(-1 if args.device == "cpu" else args.device_id),
        return_timestamps=True,
        chunk_length_s=(30 if args.long_audio else 0),
    )
    return processor, model, pipeline


def get_hf_ort(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    PRECISION[args.precision][0]
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device

    if args.hf_ort_model_path == "":
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            from_transformers=True,
            use_io_binding=(args.device == "cuda"),
        ).to(target_device)
    else:
        assert os.path.exists(args.hf_ort_model_path)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.hf_ort_model_path,
            use_io_binding=(args.device == "cuda"),
        ).to(target_device)

    pipeline = ort_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=(-1 if args.device == "cpu" else args.device_id),
        return_timestamps=True,
        chunk_length_s=(30 if args.long_audio else 0),
    )
    return processor, model, pipeline


def get_ort_model(args):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = args.profile
    sess_options.register_custom_ops_library(get_library_path())
    if args.verbose:
        sess_options.log_verbosity_level = 3
        sess_options.log_severity_level = 1
    logger.info(f"Loading model from {args.ort_model_path}")
    start_time = time.time()
    sess = ort.InferenceSession(args.ort_model_path, sess_options, providers=[args.execution_provider])
    end_time = time.time()
    logger.info(f"Loaded model in {end_time - start_time} s")
    return sess


def time_fn(args, fn, inputs):
    init_range = range(args.warmup_runs) if args.benchmark_type == "ort" else trange(args.warmup_runs, file=sys.stdout)
    inf_range = range(args.num_runs) if args.benchmark_type == "ort" else trange(args.num_runs, file=sys.stdout)

    # Warm up
    outputs = None
    for _ in init_range:
        outputs = fn(inputs)

    if args.verbose:
        logger.info(outputs)

    # Benchmark
    if args.device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in inf_range:
        outputs = fn(inputs)

    if args.device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    throughput = args.batch_size / latency
    metrics = (args.batch_size, latency, throughput)

    # Newline print after trange in order to print metrics on new line without progress bar on same line
    if args.benchmark_type != "ort":
        logger.info("\n")

    return outputs, metrics


# Benchmark types: HF + PT, HF + PT2, HF + ORT
def run_hf_pipeline_inference(args, audio, pipe):
    if args.profile:
        # Profile kernels
        with profile(  # noqa: SIM117
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                pipe([audio] * args.batch_size)
        prof_data = prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)

        # Filename format example: "hf_pt2_pipeline_<current-time>.txt"
        filename = f"{args.benchmark_type.lower().replace(' + ', '_')}_pipeline_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.txt"
        with open(filename, "w") as f:
            f.write(prof_data)

        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        pipe([audio] * args.batch_size)
        logger.info(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: pipe([audio] * args.batch_size))

        return

    transcription_fn = lambda audio: pipe([audio] * args.batch_size)  # noqa: E731
    transcription, metrics = time_fn(args, transcription_fn, audio)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")


# Benchmark types: HF + PT, HF + PT2, HF + ORT
def run_hf_generate_and_decode_inference(args, inputs, processor, model):
    def gen_and_dec():
        predicted_ids = model.generate(**inputs)
        transcription = []
        for bs in range(args.batch_size):
            for rs in range(args.num_return_sequences):
                transcription.append(
                    processor.batch_decode(
                        predicted_ids[bs * args.num_return_sequences + rs], skip_special_tokens=True
                    )[0]
                )

    if args.profile:
        # Profile kernels
        with profile(  # noqa: SIM117
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                gen_and_dec()
        prof_data = prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)

        # Filename format example: "hf_pt2_gen_and_dec_<current-time>.txt"
        filename = f"{args.benchmark_type.lower().replace(' + ', '_')}_gen_and_dec_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.txt"
        with open(filename, "w") as f:
            f.write(prof_data)

        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        gen_and_dec()
        logger.info(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: gen_and_dec())

        return

    transcription_fn = lambda dummy_input: gen_and_dec()  # noqa: E731
    transcription, metrics = time_fn(args, transcription_fn, None)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")


# Benchmark types: ORT only
def run_ort_only_inference(args, inputs, model):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        logger.error(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them to `get_ort_inputs`.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            logger.info(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Add IO binding for non-CPU inputs
    if args.device != "cpu":
        io_binding = model.io_binding()
        for k, v in inputs.items():
            io_binding.bind_cpu_input(k, v)
        for output in model.get_outputs():
            io_binding.bind_output(output.name)

    if args.profile:
        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        model.run(None, inputs) if args.device == "cpu" else model.run_with_iobinding(io_binding)
        logger.info(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Turn profiling off to stop generating logs
        args.profile = False
        model.end_profiling()

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: model.run(None, inputs))

        return

    if args.device == "cpu":
        generate_fn = lambda inputs: model.run(None, inputs)  # noqa: E731
        outputs, metrics = time_fn(args, generate_fn, inputs)
    else:
        generate_fn = lambda io_binding: model.run_with_iobinding(io_binding)  # noqa: E731
        outputs, metrics = time_fn(args, generate_fn, io_binding)
    logger.info(f"Batch size = {metrics[0]}, latency = {metrics[1]} s, throughput = {metrics[2]} qps")


def run_inference(args, inputs, processor, model, pipeline):
    if args.hf_api == "pipeline":
        run_hf_pipeline_inference(args, inputs, pipeline)
    elif args.hf_api == "gen-and-dec":
        run_hf_generate_and_decode_inference(args, inputs, processor, model)
    else:
        run_ort_only_inference(args, inputs, model)


def parse_args():
    parser = argparse.ArgumentParser()

    # Args for benchmark type
    parser.add_argument(
        "-bt", "--benchmark-type", type=str, required=True, choices=["HF + PT", "HF + PT2", "HF + ORT", "ORT"]
    )
    parser.add_argument(
        "--hf-api",
        type=str,
        choices=["pipeline", "gen-and-dec"],
        help="Whether to use Hugging Face's 'pipeline()' API or \
                        'model.generate() + processor.batch_decode()' API",
    )

    # Args for audio file and batch size
    parser.add_argument("-a", "--audio-path", type=str, default="", help="Path to audio file for E2E evaluation")
    parser.add_argument(
        "--long-audio", default=False, action="store_true", help="Whether the audio file is longer than 30s"
    )
    parser.add_argument("-b", "--batch-size", required=True, type=int, default=1)

    # Args for choosing the model
    parser.add_argument(
        "-s",
        "--model-size",
        required=True,
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
    )
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int8", "fp16", "fp32"],
        help="Precision for model and inputs. For PyTorch models, this sets the model's precision. \
                              For ONNX models, the model's precision should be set before running this script.",
    )
    parser.add_argument(
        "--hf-pt-model-path",
        type=str,
        default="",
        help="Path to directory containing all PyTorch files (e.g. tokenizer, PyTorch model)",
    )
    parser.add_argument(
        "--hf-ort-model-path",
        type=str,
        default="",
        help="Path to directory containing all ONNX files (e.g. tokenizer, encoder, decoder, decoder_with_past)",
    )
    parser.add_argument("--ort-model-path", type=str, default="", help="Path to ONNX model")

    ######################################################################################
    # Args for ORT-only benchmarking

    # Args for ORT E2E and beam search decoding
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--decoder-input-ids",
        type=list,
        default=[50258],
        help="The forced decoder ids for generation. Format is [start token, timestamp token, \
                              language token, task token]. Default is [start token]",
    )
    parser.add_argument(
        "--logits-processor",
        type=int,
        default=1,
        help="Type of logits processor to use. See `BeamSearch` in \
                        https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/contrib_ops/contrib_defs.cc \
                        for details.",
    )

    parser.add_argument(
        "-f", "--feature-size", type=int, default=80, help="Known as 'feature size' or 'number of Mels'"
    )
    parser.add_argument(
        "-es",
        "--encoder-seq-len",
        type=int,
        default=3000,
        help="Known as 'encoder sequence length' or 'number of frames'",
    )

    # When skipping pre/post processing and not evaluating E2E (e.g. ORT component-wise),
    # the following input args can be used:

    # Dynamic inputs:
    parser.add_argument(
        "-ds", "--decoder-seq-len", type=int, default=448, help="Maximum decoder sequence length is 448"
    )
    parser.add_argument(
        "-pds", "--past-decoder-seq-len", type=int, default=447, help="Maximum past decoder sequence length is 447"
    )
    ######################################################################################

    # Args for running and evaluating the model
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "rocm"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0, help="Device ID when using GPU")
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-r", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Args for accessing detailed info
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Whether to profile the model (e.g. CPU usage, memory footprint)",
    )
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print information (e.g. outputs, verifications)",
    )

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    if "ORT" in args.benchmark_type:
        args.execution_provider = f"{args.device.upper()}ExecutionProvider"
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
        elif args.execution_provider == "ROCMExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
            args.device = "cuda"

    if args.benchmark_type == "ORT":
        args.hf_api = None

    # Set model properties
    setattr(args, "model_name", f"openai/whisper-{args.model_size}")  # noqa: B010
    config = AutoConfig.from_pretrained(args.model_name)
    (num_layers, num_heads, hidden_size) = config.num_hidden_layers, config.decoder_attention_heads, config.d_model
    setattr(args, "num_layers", num_layers)  # noqa: B010
    setattr(args, "num_heads", num_heads)  # noqa: B010
    setattr(args, "hidden_size", hidden_size)  # noqa: B010
    setattr(args, "head_size", hidden_size // num_heads)  # noqa: B010

    return args


def main():
    args = parse_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True
    inputs, processor, model, pipeline = get_vars(args)
    run_inference(args, inputs, processor, model, pipeline)


if __name__ == "__main__":
    main()
