import hashlib
import json
from pathlib import Path
import sys

import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_webgpu_fp8_checkpoint import natural_text_prompt, token_hash


root = Path(__file__).resolve().parent
model = Path("/workspace/qwen38-nvfp4-webgpu")
manifest = json.loads((root / "natural-pool-manifest.json").read_text())
text = (root / "alice-natural-pool.txt").read_bytes()
if hashlib.sha256(text).hexdigest() != manifest["pool_sha256"]:
    raise ValueError("Text pool differs from manifest")
if hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
    raise ValueError("Tokenizer differs from manifest")
tokens = og.Tokenizer(str(model)).encode(text.decode("utf-8"))
if len(tokens) != manifest["pool_tokens"] or token_hash(tokens) != manifest["tokens_sha256"]:
    raise ValueError("GenAI and CPU token pools differ")
for row in manifest["prompts"]:
    prompt, offset = natural_text_prompt(tokens, row["context_tokens"], row["seed"])
    if offset != row["offset"] or token_hash(prompt) != row["prompt_sha256"]:
        raise ValueError(f"Frozen prompt mismatch: {row}")
result = {"status": "passed", "tokenizer": "actual GenAI CPU tokenizer without model load",
          "pool_tokens": len(tokens), "prompts_verified": len(manifest["prompts"]),
          "tokens_sha256": token_hash(tokens)}
(root / "genai-pool-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
