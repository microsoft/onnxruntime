import hashlib
import json
import os
from pathlib import Path

source = Path("/home/sayanshaw/repos/qwen_38_flash_nvfp4_engine_indexshare")
destination = Path("/home/sayanshaw/repos/qwen38-nvfp4-webgpu")
destination.mkdir()
for file in source.iterdir():
    if file.is_file() and file.name != "genai_config.json":
        os.link(file, destination / file.name)
config_path = source / "genai_config.json"
config = json.loads(config_path.read_text())
for section in ("decoder", "mtp"):
    config["model"][section]["session_options"] = {
        "provider_options": [{"webgpu": {
            "enableGraphCapture": "0", "validationMode": "disabled", "enableInt64": "1",
        }}],
        "session.use_device_allocator_for_initializers": "1",
        "session.disable_cpu_ep_fallback": "1",
        "log_severity_level": 3,
    }
config["model"]["engram"]["session_options"] = {"provider_options": [{"cpu": {}}]}
original_index_share = config["model"]["mtp"].pop("index_share")
config["model"]["mtp"]["enabled"] = False
config["engine"]["dynamic_batching"] = {"block_size": 256, "max_batch_size": 1, "num_blocks": 160}
config["search"]["do_sample"] = False
config["search"]["chunk_size"] = 16
(destination / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
(destination / "webgpu-overlay.json").write_text(json.dumps({
    "source": str(source),
    "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "quantization": "Original NVFP4 target and FP8 MTP bytes, unmodified hardlinks",
    "main_graph_sha256": hashlib.sha256((source / "model.onnx").read_bytes()).hexdigest(),
    "changes": [
        "WebGPU decoder with CPU fallback forbidden",
        "CPU Engram without CUDA host registration",
        "Explicit 160-block cache budget, checked against actual Engine capacity before requests",
        "Greedy decoding, 16-token prefill chunks",
        "MTP disabled for initial target-only qualification: packed index-share graph requires adaptation",
    ],
    "original_mtp_index_share": original_index_share,
    "not_tianlei_v12b": True,
}, indent=2) + "\n")
print(destination)
