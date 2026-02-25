"""
Stable Diffusion Component Latency Benchmark

Benchmarks latency of SD pipeline component models for two sets:
  - float (dynamic shapes)
  - qdq (quantized, mostly static shapes)

Only directories that contain a `model.onnx` are benchmarked.

Output is a nested dictionary structure:
{
  "latency_ms": {
	"float": {
	  "unet": { ...stats... },
	  "text_encoder": { ...stats... }
	},
	"qdq": {
	  "unet": { ...stats... },
	  ...
	}
  },
  "metadata": { ... }
}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
	import onnxruntime as ort
except ImportError:
	print("Error: onnxruntime not installed. Run: pip install onnxruntime")
	sys.exit(1)


DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 5


DYNAMIC_DIM_MAP: Dict[str, int] = {
	"batch": 1,
	"sequence": 77,
	"unet_sample_batch": 1,
	"unet_sample_channels": 4,
	"unet_sample_height": 64,
	"unet_sample_width": 64,
	"unet_time_batch": 1,
	"unet_hidden_batch": 1,
	"unet_hidden_sequence": 77,
	"decoder_batch": 1,
	"decoder_channels": 4,
	"decoder_height": 64,
	"decoder_width": 64,
	"encoder_batch": 1,
	"encoder_channels": 3,
	"encoder_height": 512,
	"encoder_width": 512,
	"Addlatent_sample_dim_0": 1,
	"Addlatent_sample_dim_1": 4,
	"Addlatent_sample_dim_2": 64,
	"Addlatent_sample_dim_3": 64,
}


@dataclass
class BenchmarkConfig:
	model_root: str
	warmup: int = DEFAULT_WARMUP
	iterations: int = DEFAULT_ITERATIONS
	output_file: Optional[str] = None
	seed: int = 42
	verbose: bool = False


def inspect_model_inputs(session: ort.InferenceSession) -> List[Dict[str, Any]]:
	inputs = []
	for inp in session.get_inputs():
		inputs.append(
			{
				"name": inp.name,
				"shape": inp.shape,
				"type": inp.type,
			}
		)
	return inputs


def numpy_dtype_from_onnx_type(onnx_type: str) -> np.dtype:
	type_map = {
		"tensor(float)": np.float32,
		"tensor(float16)": np.float16,
		"tensor(double)": np.float64,
		"tensor(int64)": np.int64,
		"tensor(int32)": np.int32,
		"tensor(int16)": np.int16,
		"tensor(int8)": np.int8,
		"tensor(uint8)": np.uint8,
		"tensor(bool)": np.bool_,
	}
	return type_map.get(onnx_type, np.float32)


def resolve_dynamic_dim(
	dim: Any,
	input_name: str,
	dynamic_dim_map: Dict[str, int],
) -> int:
	if isinstance(dim, int):
		return dim

	if dim is None:
		dim_str = ""
	else:
		dim_str = str(dim)

	if dim_str in dynamic_dim_map:
		return dynamic_dim_map[dim_str]

	dim_lower = dim_str.lower()

	if dim_lower in dynamic_dim_map:
		return dynamic_dim_map[dim_lower]

	raise ValueError(
		f"Unknown dynamic dim '{dim_str or '<None>'}' for input '{input_name}'. "
		"Add it to DYNAMIC_DIM_MAP."
	)


def resolve_shape(
	shape: Sequence[Any],
	input_name: str,
	dynamic_dim_map: Dict[str, int],
) -> List[int]:
	resolved: List[int] = []
	for dim in shape:
		resolved.append(resolve_dynamic_dim(dim, input_name, dynamic_dim_map))
	return resolved


def create_input_tensor(input_name: str, dtype: np.dtype, shape: List[int]) -> np.ndarray:
	name_lower = input_name.lower()

	if np.issubdtype(dtype, np.integer):
		if "input_ids" in name_lower:
			return np.random.randint(0, 49408, size=shape, dtype=dtype)
		return np.zeros(shape, dtype=dtype)

	if np.issubdtype(dtype, np.bool_):
		return np.ones(shape, dtype=dtype)

	return np.random.randn(*shape).astype(dtype)


def generate_inputs(
	session: ort.InferenceSession,
	dynamic_dim_map: Dict[str, int],
) -> Dict[str, np.ndarray]:
	inputs: Dict[str, np.ndarray] = {}

	for info in inspect_model_inputs(session):
		input_name = info["name"]
		resolved_shape = resolve_shape(
			info["shape"], input_name, dynamic_dim_map
		)
		dtype = numpy_dtype_from_onnx_type(info["type"])
		inputs[input_name] = create_input_tensor(input_name, dtype, resolved_shape)

	return inputs


def create_session(model_path: str) -> ort.InferenceSession:
	sess_options = ort.SessionOptions()
	return ort.InferenceSession(
		model_path,
		sess_options,
		providers=["CPUExecutionProvider"],
	)


def run_latency(
	session: ort.InferenceSession,
	inputs: Dict[str, np.ndarray],
	warmup: int,
	iterations: int,
) -> List[float]:
	output_names = [o.name for o in session.get_outputs()]

	for _ in range(warmup):
		_ = session.run(output_names, inputs)

	latencies_ms: List[float] = []
	for _ in range(iterations):
		start = time.perf_counter()
		_ = session.run(output_names, inputs)
		end = time.perf_counter()
		latencies_ms.append((end - start) * 1000.0)

	return latencies_ms


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
	arr = np.array(latencies_ms, dtype=np.float64)
	return {
		"mean_ms": round(float(np.mean(arr)), 3),
		"std_ms": round(float(np.std(arr)), 3),
		"min_ms": round(float(np.min(arr)), 3),
		"max_ms": round(float(np.max(arr)), 3),
		"p50_ms": round(float(np.percentile(arr, 50)), 3),
		"p95_ms": round(float(np.percentile(arr, 95)), 3),
		"p99_ms": round(float(np.percentile(arr, 99)), 3),
	}


def find_component_models(set_dir: str) -> List[Tuple[str, str]]:
	"""
	Recursively find component folders that contain model.onnx.

	Returns list of tuples: (component_key, model_path)
	where component_key is relative folder path from set_dir.
	"""
	found: List[Tuple[str, str]] = []
	for root, _, files in os.walk(set_dir):
		if "model.onnx" in files:
			component_key = os.path.relpath(root, set_dir).replace("\\", "/")
			found.append((component_key, os.path.join(root, "model.onnx")))

	found.sort(key=lambda item: item[0])
	return found


def benchmark_component(
	set_name: str,
	component_key: str,
	model_path: str,
	warmup: int,
	iterations: int,
	dynamic_dim_map: Dict[str, int],
	verbose: bool,
) -> Dict[str, Any]:
	if verbose:
		print(f"  [{set_name}] {component_key}: loading {model_path}")

	session = create_session(model_path)
	inputs = generate_inputs(session, dynamic_dim_map)
	latencies = run_latency(session, inputs, warmup, iterations)
	stats = latency_stats(latencies)

	result: Dict[str, Any] = {
		"iterations": iterations,
		"warmup": warmup,
		"model_path": model_path,
		"inputs": {
			k: {
				"shape": list(v.shape),
				"dtype": str(v.dtype),
			}
			for k, v in inputs.items()
		},
	}
	result.update(stats)
	return result


def benchmark_all_components(config: BenchmarkConfig) -> Dict[str, Any]:
	root = os.path.abspath(config.model_root)
	sets = ["float", "qdq"]

	output: Dict[str, Any] = {
		"latency_ms": {"float": {}, "qdq": {}},
		"metadata": {
			"model_root": root,
			"warmup": config.warmup,
			"iterations": config.iterations,
			"seed": config.seed,
			"ort_version": ort.__version__,
			"platform": platform.platform(),
			"python_version": platform.python_version(),
		},
	}

	for set_name in sets:
		set_dir = os.path.join(root, set_name)
		if not os.path.isdir(set_dir):
			print(f"Warning: set folder not found, skipping: {set_dir}")
			continue

		components = find_component_models(set_dir)
		if not components:
			print(f"Warning: no model.onnx found under: {set_dir}")
			continue

		print(f"\nBenchmarking set '{set_name}' ({len(components)} components)")
		for component_key, model_path in components:
			print(f"- {component_key}")
			component_result = benchmark_component(
				set_name=set_name,
				component_key=component_key,
				model_path=model_path,
				warmup=config.warmup,
				iterations=config.iterations,
				dynamic_dim_map=DYNAMIC_DIM_MAP,
				verbose=config.verbose,
			)
			output["latency_ms"][set_name][component_key] = component_result

	return output


def save_results(data: Dict[str, Any], output_file: str) -> str:
	output_file = os.path.abspath(output_file)
	output_dir = os.path.dirname(output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)

	return output_file


def default_output_path(model_root: str) -> str:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	model_name = os.path.basename(os.path.abspath(model_root.rstrip("\\/")))
	if not model_name:
		model_name = "sd"
	return os.path.join(script_dir, "results", f"{model_name}_components_latency.json")


def parse_args() -> BenchmarkConfig:
	parser = argparse.ArgumentParser(
		description="Benchmark SD component latency for float and qdq model sets"
	)
	parser.add_argument(
		"--model-root",
		"-m",
		required=True,
		help="Path to SD root folder that contains 'float' and 'qdq' subfolders",
	)
	parser.add_argument(
		"--warmup",
		"-w",
		type=int,
		default=DEFAULT_WARMUP,
		help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
	)
	parser.add_argument(
		"--iterations",
		"-i",
		type=int,
		default=DEFAULT_ITERATIONS,
		help=f"Benchmark iterations (default: {DEFAULT_ITERATIONS})",
	)
	parser.add_argument(
		"--output",
		"-o",
		default=None,
		help="Output JSON path (default: results/<model_root_name>_components_latency.json)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility (default: 42)",
	)
	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Verbose logging",
	)

	args = parser.parse_args()
	return BenchmarkConfig(
		model_root=args.model_root,
		warmup=args.warmup,
		iterations=args.iterations,
		output_file=args.output,
		seed=args.seed,
		verbose=args.verbose,
	)


def main() -> None:
	config = parse_args()

	if not os.path.isdir(config.model_root):
		print(f"Error: model root folder not found: {config.model_root}")
		sys.exit(1)

	np.random.seed(config.seed)

	start = time.perf_counter()
	results = benchmark_all_components(config)
	elapsed_s = time.perf_counter() - start

	results["metadata"]["total_elapsed_s"] = round(elapsed_s, 3)

	output_path = config.output_file or default_output_path(config.model_root)
	saved_path = save_results(results, output_path)

	print("\nDone!")
	print(f"Saved: {saved_path}")


if __name__ == "__main__":
	main()
