#!/usr/bin/env python3
"""Run a list of prompts with ONNX Runtime GenAI and collect MoE routing logs."""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import onnxruntime_genai as og


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Directory containing genai_config.json.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompts-file",
        type=Path,
        help="JSON array or JSON Lines file containing prompts.",
    )
    prompt_group.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to run; repeat this option to submit several prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qmoe-prompt-results.json"),
        help="Generated-text and timing output.",
    )
    parser.add_argument(
        "--routing-log",
        type=Path,
        default=Path("qmoe-routing.log"),
        help="Native ORT log receiving the MoE routing records.",
    )
    parser.add_argument(
        "--provider",
        choices=("cuda", "cpu", "follow_config"),
        default="cuda",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--raw-prompts",
        action="store_true",
        help="Do not apply the tokenizer chat template.",
    )
    return parser.parse_args()


def load_prompts(path):
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list) or not data:
        raise ValueError("The prompts file must contain a non-empty JSON list.")

    prompts = []
    for index, item in enumerate(data, start=1):
        prompt = item.get("prompt") if isinstance(item, dict) else item
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Prompt {index} must be a non-empty string.")
        prompts.append(prompt)
    return prompts


@contextmanager
def redirect_native_stderr(path):
    """Redirect Python and native-library stderr to one file descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_stderr = os.dup(sys.stderr.fileno())
    with path.open("w", encoding="utf-8") as stream:
        sys.stderr.flush()
        os.dup2(stream.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stderr)


def create_model(model_path, provider):
    config = og.Config(str(model_path))
    if provider != "follow_config":
        config.clear_providers()
        if provider != "cpu":
            config.append_provider(provider)
    config.overlay(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "session_options": {
                            "log_severity_level": 1,
                            "session.enable_moe_expert_statistics": "1",
                        }
                    }
                }
            }
        )
    )
    return og.Model(config)


def format_prompt(tokenizer, prompt, raw_prompt):
    if raw_prompt:
        return prompt
    messages = json.dumps([{"role": "user", "content": prompt}])
    return tokenizer.apply_chat_template(
        messages=messages,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt, max_new_tokens, raw_prompt):
    formatted_prompt = format_prompt(tokenizer, prompt, raw_prompt)
    prompt_tokens = tokenizer.encode(formatted_prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(prompt_tokens) + max_new_tokens,
        do_sample=False,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(prompt_tokens)

    generated_tokens = []
    start = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
        generated_tokens.extend(generator.get_next_tokens().tolist())
    duration = time.perf_counter() - start

    return {
        "generated_text": tokenizer.decode(generated_tokens),
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": len(generated_tokens),
        "duration_seconds": duration,
        "tokens_per_second": len(generated_tokens) / duration,
    }


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    prompts = (
        load_prompts(args.prompts_file)
        if args.prompts_file
        else args.prompts
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with redirect_native_stderr(args.routing_log):
        model = create_model(args.model, args.provider)
        tokenizer = og.Tokenizer(model)
        results = []
        for prompt_index, prompt in enumerate(prompts, start=1):
            print(
                f"[qmoe_prompt_runner] {prompt_index}/{len(prompts)} prompt_start",
                file=sys.stderr,
                flush=True,
            )
            result = generate(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.raw_prompts,
            )
            results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    **result,
                }
            )
            print(
                f"[{prompt_index}/{len(prompts)}] "
                f"{result['generated_tokens']} tokens in "
                f"{result['duration_seconds']:.3f}s"
            )

    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(args.routing_log)


if __name__ == "__main__":
    main()
