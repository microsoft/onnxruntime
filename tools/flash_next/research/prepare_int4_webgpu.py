import hashlib
import json
import os
from pathlib import Path


source = Path("/home/sayanshaw/repos/qwen38-webgpu-raw")
destination = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-perf")
destination.mkdir(exist_ok=True)
if any(destination.iterdir()):
    raise FileExistsError(f"Refusing to overwrite nonempty overlay: {destination}")
for file in source.iterdir():
    if file.is_file() and file.name != "genai_config.json":
        os.link(file, destination / file.name)
config_path = source / "genai_config.json"
config = json.loads(config_path.read_text())
options = config["model"]["decoder"]["session_options"]
options["intra_op_num_threads"] = 1
options["inter_op_num_threads"] = 1
options["provider_options"] = [{"webgpu": {
    "enableGraphCapture": "0", "validationMode": "disabled", "enableInt64": "1",
}}]
config["search"]["do_sample"] = False
config["search"]["chunk_size"] = 16
config["search"]["past_present_share_buffer"] = True
(destination / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
(destination / "webgpu-overlay.json").write_text(json.dumps({
    "source": str(source),
    "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "graph_sha256": hashlib.sha256((source / "model.onnx").read_bytes()).hexdigest(),
    "quantization": "Original INT4 RTN MoE block32, INT4 dense and INT8 final matmul; unchanged source bytes",
    "runtime": "Generator, no MTP, non-paged attention, explicit CPU embedding layer retained",
    "changes": ["WebGPU validation disabled for timing; graph capture remains disabled",
                "Greedy decoding with explicit 16-token prefill chunks",
                "Shared attention KV retained; GenAI handles legacy growing indexer caches independently",
                "One intra-op and inter-op host thread for the GPU-heavy decoder"],
    "not_tianlei_v12b": True,
}, indent=2) + "\n")
print(destination)
