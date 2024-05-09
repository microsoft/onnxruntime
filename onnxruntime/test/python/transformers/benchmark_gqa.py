import math
from typing import Optional

import torch
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager


class AttentionConfig:
    def __init__(
        self,
        operator: str,
        batch_size: int,
        sequence_length: int,
        max_sequence_length: int,
        past_sequence_length: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        softmax_scale: Optional[float],
        do_rotary: bool,
        rotary_interleaved: bool,
        device="cuda",
        dtype=torch.float16,
        share_buffer: bool = True,
        is_packed_qkv: bool = False,
    ):
        self.operator = operator
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_size**0.5)

        # Derived values
        self.total_sequence_length = sequence_length + past_sequence_length
        self.past_buffer_length = max_sequence_length if share_buffer else past_sequence_length
        self.present_buffer_length = max_sequence_length if share_buffer else (past_sequence_length + sequence_length)

        self.do_rotary = do_rotary
        self.rotary_interleaved = rotary_interleaved
        self.device = device

        self.share_buffer = share_buffer
        self.is_packed_qkv = is_packed_qkv
        self.dtype = dtype

    def shape_dict(self):
        return {
            "query": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "key": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "value": (self.batch_size, self.sequence_length, self.kv_num_heads * self.head_size),
            "past_key": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "past_value": (self.batch_size, self.kv_num_heads, self.past_buffer_length, self.head_size),
            "total_sequence_length": (1,),
            "output": (self.batch_size, self.sequence_length, self.num_heads * self.head_size),
            "present_key": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "present_value": (self.batch_size, self.kv_num_heads, self.present_buffer_length, self.head_size),
            "cos_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
            "sin_cache": (self.max_sequence_length, (math.floor(self.head_size / 16) * 16) // 2),
        }

    def get_cos_sin_cache(self, dtype):
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * self.head_size) / 16) * 16
        angle = torch.rand(self.max_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        return cos.to(device=self.device), sin.to(device=self.device)

    def random_inputs(self):
        device = self.device
        # bfloat16 is not supported in ORT python I/O binding API
        dtype = torch.float16
        shape_dict = self.shape_dict()

        torch.manual_seed(123)
        feeds = {
            "query": torch.empty(shape_dict["query"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "key": torch.empty(shape_dict["key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "value": torch.empty(shape_dict["value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_key": torch.empty(shape_dict["past_key"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "past_value": torch.empty(shape_dict["past_value"], device=device, dtype=dtype).normal_(mean=0, std=0.1),
            "total_sequence_length": torch.tensor([self.total_sequence_length], dtype=torch.int32),
        }

        if self.do_rotary:
            cos_cache, sin_cache = self.get_cos_sin_cache(dtype)
            feeds["cos_cache"] = cos_cache
            feeds["sin_cache"] = sin_cache

        return feeds


class GroupQueryAttentionConfig(AttentionConfig):
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        max_sequence_length: int,
        past_sequence_length: int,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        softmax_scale=None,
        do_rotary: bool = False,
        rotary_interleaved: bool = False,
        device="cuda",
        local_window_size: int = -1,
    ):
        super().__init__(
            "GroupQueryAttention",
            batch_size,
            sequence_length,
            max_sequence_length,
            past_sequence_length,
            num_heads,
            kv_num_heads,
            head_size,
            softmax_scale,
            do_rotary,
            rotary_interleaved,
            device,
        )
        self.local_window_size = local_window_size

    def shape_dict(self):
        shapes = super().shape_dict()
        shapes.update(
            {
                "seqlens_k": (self.batch_size,),
            }
        )
        return shapes

    def random_inputs(self):
        feeds = super().random_inputs()
        k_seqlens = torch.ones((self.batch_size,), device=self.device, dtype=torch.int32) * self.total_sequence_length
        feeds.update(
            {
                "seqlens_k": k_seqlens - 1,
            }
        )
        return feeds


def create_group_query_attention_onnx_model(config: GroupQueryAttentionConfig):
    assert config.dtype == torch.float16

    float_type = TensorProto.FLOAT16
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not config.is_packed_qkv else "",
                "value" if not config.is_packed_qkv else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length" if config.share_buffer else "",
                "cos_cache" if config.do_rotary else "",
                "sin_cache" if config.do_rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            scale=config.softmax_scale,
            local_window_size=config.local_window_size,
            do_rotary=1 if config.do_rotary else 0,
            rotary_interleaved=config.rotary_interleaved,
            domain="com.microsoft",
        ),
    ]

    shape_dict = config.shape_dict()
    graph_input = [
        helper.make_tensor_value_info("query", float_type, list(shape_dict["query"])),
        helper.make_tensor_value_info("key", float_type, list(shape_dict["key"])),
        helper.make_tensor_value_info("value", float_type, list(shape_dict["value"])),
        helper.make_tensor_value_info("past_key", float_type, list(shape_dict["past_key"])),
        helper.make_tensor_value_info("past_value", float_type, list(shape_dict["past_value"])),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, list(shape_dict["seqlens_k"])),
        helper.make_tensor_value_info(
            "total_sequence_length", TensorProto.INT32, list(shape_dict["total_sequence_length"])
        ),
    ]

    if config.do_rotary:
        graph_input += [
            helper.make_tensor_value_info("cos_cache", float_type, list(shape_dict["cos_cache"])),
            helper.make_tensor_value_info("sin_cache", float_type, list(shape_dict["sin_cache"])),
        ]

    graph_output = [
        helper.make_tensor_value_info("output", float_type, list(shape_dict["output"])),
        helper.make_tensor_value_info("present_key", float_type, list(shape_dict["present_key"])),
        helper.make_tensor_value_info("present_value", float_type, list(shape_dict["present_value"])),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_session(onnx_model_str, cuda_provider_options=None) -> InferenceSession:
    session_options = SessionOptions()
    ort_session = InferenceSession(
        onnx_model_str,
        session_options,
        providers=[("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"],
    )
    return ort_session


class OrtGroupQueryAttention:
    """A wrapper of ORT GroupQueryAttention to test relevance and performance."""

    def __init__(self, config: GroupQueryAttentionConfig):
        device = config.device
        cuda_provider_options = CudaSession.get_cuda_provider_options(
            torch.cuda.current_device(), enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
        )
        onnx_model_str = create_group_query_attention_onnx_model(config)
        self.ort_session = create_session(onnx_model_str, cuda_provider_options=cuda_provider_options)
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=self.ort_session,
            device=device,
            stream=torch.cuda.current_stream().cuda_stream,
            max_cuda_graphs=2,
        )
        buffer_sharing = {"past_key": "present_key", "past_value": "present_value"}
        self.gpu_binding = self.gpu_binding_manager.get_binding(
            config.shape_dict(), use_cuda_graph=False, buffer_sharing=buffer_sharing
        )
        self.feed_dict = config.random_inputs()

    def infer(self):
        return self.gpu_binding.infer(self.feed_dict)


def get_plot_algos(sm: int):
    # GQA with local windows only works in sm=8x
    if sm >= 80:
        return {
            "line_vals": ["ort_gqa", "ort_gqa_local"],
            "line_names": ["ORT-GQA-Dense", "ORT-GQA-Local"],
            "styles": [("red", "-"), ("blue", "-")],
        }
    else:
        return {
            "line_vals": ["ort_gqa"],
            "line_names": ["ORT-GQA-Dense"],
            "styles": [("green", "-")],
        }


def plot_prompt_performance(
    sm: int,
    batch_size=4,
    num_heads=32,
    kv_num_heads=8,
    max_seq_len=8192,
    head_size=128,
):
    import triton

    algos = get_plot_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["sequence_length"],
            x_vals=[2**i for i in range(4, 14)],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"prompt-sm{sm}-batch{batch_size}-head{num_heads}_kv{kv_num_heads}-d{head_size}-fp16",
            args={
                "num_heads": num_heads,
                "kv_num_heads": kv_num_heads,
                "batch_size": batch_size,
                "head_size": head_size,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(batch_size, num_heads, kv_num_heads, sequence_length, head_size, provider, device="cuda"):
        warmup = 15
        repeat = 100

        config: GroupQueryAttentionConfig = GroupQueryAttentionConfig(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_sequence_length=max_seq_len,
            past_sequence_length=0,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            local_window_size=1024 if provider == "ort_gqa_local" else -1,
            device=device,
        )

        obj = OrtGroupQueryAttention(config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def plot_token_performance(
    sm: int,
    batch_size=4,
    num_heads=32,
    kv_num_heads=8,
    max_seq_len=8192,
    head_size=128,
):
    import triton

    algos = get_plot_algos(sm)
    configs = [
        triton.testing.Benchmark(
            x_names=["past_sequence_length"],
            x_vals=[2**i for i in range(4, 13)] + [max_seq_len - 1],
            line_arg="provider",
            ylabel="ms",
            **algos,
            plot_name=f"token-sm{sm}-batch{batch_size}-head{num_heads}_kv{kv_num_heads}-d{head_size}-fp16",
            args={
                "num_heads": num_heads,
                "kv_num_heads": kv_num_heads,
                "batch_size": batch_size,
                "head_size": head_size,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(
        batch_size,
        num_heads,
        kv_num_heads,
        past_sequence_length,
        head_size,
        provider,
        device="cuda",
    ):
        warmup = 15
        repeat = 100

        config: GroupQueryAttentionConfig = GroupQueryAttentionConfig(
            batch_size=batch_size,
            sequence_length=1,
            max_sequence_length=max_seq_len,
            past_sequence_length=past_sequence_length,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            local_window_size=1024 if provider == "ort_gqa_local" else -1,
            device=device,
        )

        obj = OrtGroupQueryAttention(config)

        ms = triton.testing.do_bench(obj.infer, warmup=warmup, rep=repeat)
        return ms

    benchmark.run(save_path=".", print_data=True)


def run_performance_test(sm: int):
    """
    Run performance tests for prompt and token generation.

    """
    for batch_size in [1, 4, 8, 16]:
        for num_heads, kv_num_heads in [(8, 8), (16, 8), (32, 8), (64, 8)]:
            for head_size in [64, 128]:
                plot_prompt_performance(
                    sm=sm,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    max_seq_len=8192,
                    head_size=head_size,
                )
                plot_token_performance(
                    sm=sm,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    max_seq_len=8192,
                    head_size=head_size,
                )


if __name__ == "__main__":
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.no_grad():
        run_performance_test(sm)
