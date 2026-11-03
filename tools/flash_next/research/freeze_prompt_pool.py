import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("webgpu_benchmark", root.parent / "benchmark_webgpu_fp8_checkpoint.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
raw_path = root / "alice-public-domain.txt"
text = raw_path.read_text()
start = "*** START OF THE PROJECT GUTENBERG EBOOK"
end = "*** END OF THE PROJECT GUTENBERG EBOOK"
if start not in text or end not in text:
    raise ValueError("Incomplete public-domain source download")
pool = text.split(start, 1)[1].split("***", 1)[1].split(end, 1)[0].strip() + "\n"
pool_path = root / "alice-natural-pool.txt"
pool_path.write_text(pool)
tokenizer_path = Path("/home/sayanshaw/repos/qwen_38_flash_nvfp4_engine_indexshare/tokenizer.json")
tokenizer = Tokenizer.from_file(str(tokenizer_path))
tokens = tokenizer.encode(pool, add_special_tokens=False).ids
if len(tokens) <= 32768:
    raise ValueError(f"Natural text pool is too short: {len(tokens)} tokens")
np.asarray(tokens, dtype="<i4").tofile(root / "alice-natural-pool.i32")
manifest = {
    "source": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "work": "Alice's Adventures in Wonderland, Lewis Carroll, 1865; public domain",
    "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
    "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    "tokens_sha256": benchmark.token_hash(tokens),
    "pool_tokens": len(tokens),
    "not_exact_tianlei_reproduction": True,
    "protocol": {"new_tokens": 256, "measured_repetitions": 3, "excluded_warmups": 1},
    "prompts": [],
}
for length in (2048, 8192, 32768):
    seeds = [1000 + length] + [i * 7 + length for i in range(3)]
    hashes = []
    for index, seed in enumerate(seeds):
        prompt, offset = benchmark.natural_text_prompt(tokens, length, seed)
        digest = benchmark.token_hash(prompt)
        if len(prompt) != length or digest in hashes:
            raise ValueError(f"Invalid or duplicate prompt at context {length}")
        hashes.append(digest)
        manifest["prompts"].append({
            "context_tokens": length, "warmup": index == 0, "seed": seed,
            "offset": offset, "prompt_sha256": digest,
        })
(root / "natural-pool-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
