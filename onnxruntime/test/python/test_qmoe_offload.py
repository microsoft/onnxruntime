"""Host checks and explicitly scheduled CUDA-plugin parity checks for QMoEOffload."""

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

GPU_CONFIG = argparse.Namespace(gpu_tests=False)


HOST_TEST_SOURCE = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <thread>
#include <vector>
#include "contrib_ops/cuda/moe/qmoe_offload_plan.h"
#include "core/providers/cuda/bounded_pinned_pool.h"

using namespace onnxruntime::contrib::cuda::qmoe;

int main() {
    int allocations = 0;
    int frees = 0;
    void* quarantined = nullptr;
    {
        onnxruntime::cuda::BoundedPinnedPool pool(128, [&](size_t bytes) {
            ++allocations;
            return std::malloc(bytes);
        }, [&](void* pointer) { ++frees; std::free(pointer); });
        auto* first = pool.Acquire(64);
        auto* second = pool.Acquire(64);
        assert(first != second && allocations == 2);
        bool exhausted = false;
        try { pool.Acquire(1); } catch (const std::runtime_error&) { exhausted = true; }
        assert(exhausted);
        pool.Release(first, true);
        assert(pool.Acquire(32) == first && allocations == 2);
        pool.Release(first, true);
        pool.Release(second, false);
        quarantined = second;
        exhausted = false;
        try { pool.Acquire(65); } catch (const std::runtime_error&) { exhausted = true; }
        assert(exhausted);
    }
    assert(frees == 1);
    std::free(quarantined);
    {
        onnxruntime::cuda::BoundedPinnedPool pool(128, [](size_t bytes) { return std::malloc(bytes); },
                                                                                        [](void* pointer) { std::free(pointer); });
        std::mutex barrier_mutex;
        std::condition_variable barrier;
        int arrived = 0;
        std::array<void*, 2> pointers{};
        auto borrow = [&](int index) {
            pointers[index] = pool.Acquire(64);
            {
                std::unique_lock<std::mutex> lock(barrier_mutex);
                ++arrived;
                barrier.notify_all();
                barrier.wait(lock, [&] { return arrived == 2; });
            }
            pool.Release(pointers[index], true);
        };
        std::thread first(borrow, 0), second(borrow, 1);
        first.join();
        second.join();
        assert(pointers[0] != pointers[1]);
    }
    for (const auto shape : {std::array<int, 2>{1280, 80}, {2560, 20}, {37, 13}, {1, 1}}) {
        const int columns = shape[0];
        const int blocks = shape[1];
        std::vector<uint16_t> source(static_cast<size_t>(columns) * blocks);
        std::vector<uint16_t> target(source.size(), 0);
        for (size_t index = 0; index < source.size(); ++index)
            source[index] = static_cast<uint16_t>(index * 1743 + 0x7c01);
        TransposeOffloadScales(source.data(), target.data(), columns, blocks);
        for (int column = 0; column < columns; ++column)
            for (int block = 0; block < blocks; ++block)
                assert(target[static_cast<size_t>(block) * columns + column] ==
                             source[static_cast<size_t>(column) * blocks + block]);
    }
  std::array<int, 512> slots{};
  std::array<int, 3> experts{};
  std::array<int, 20> routes{}, compact{}, seen{};
  const std::array<int, 20> selected{
      511, 7, 40, 5, 2, 123, 12, 30, 90, 0,
      0, 90, 30, 12, 123, 2, 5, 40, 7, 511};
  assert(InitializeOffloadRoutes(selected.data(), 20, 512, 10, slots.data()));
  int next = 0;
  int wave_count = 0;
  int route_total = 0;
  while (next < 512) {
    const auto wave = MakeOffloadWave(selected.data(), 20, 512, next, 3,
                                     slots.data(), experts.data(), routes.data(), compact.data());
    assert(wave.next_expert > next);
    next = wave.next_expert;
    if (wave.expert_count == 0) continue;
    ++wave_count;
    assert(wave.expert_count <= 3);
    assert(wave.route_count <= 2 * wave.expert_count);
    for (int index = 0; index < wave.route_count; ++index) {
      assert(compact[index] >= 0 && compact[index] < wave.expert_count);
      assert(experts[compact[index]] == selected[routes[index]]);
      assert(++seen[routes[index]] == 1);
      ++route_total;
    }
  }
  assert(wave_count == 4 && route_total == 20);
  for (const auto count : seen) assert(count == 1);
  assert(!InitializeOffloadRoutes(selected.data(), 20, 512, 0, slots.data()));
  assert(!InitializeOffloadRoutes(selected.data(), 19, 512, 10, slots.data()));
  const int invalid[] = {0, 512};
  assert(!InitializeOffloadRoutes(invalid, 2, 512, 2, slots.data()));
  const int negative[] = {0, -1};
  assert(!InitializeOffloadRoutes(negative, 2, 512, 2, slots.data()));
  const int duplicate[] = {7, 7};
  assert(!InitializeOffloadRoutes(duplicate, 2, 512, 2, slots.data()));
  const int tied_tokens[] = {0, 1, 0, 1};
  assert(InitializeOffloadRoutes(tied_tokens, 4, 512, 2, slots.data()));
  const auto first = MakeOffloadWave(tied_tokens, 4, 512, 0, 1,
                                   slots.data(), experts.data(), routes.data(), compact.data());
  assert(first.expert_count == 1 && first.route_count == 2);
  assert(routes[0] == 0 && routes[1] == 2 && compact[0] == 0 && compact[1] == 0);
}
"""


class TestQMoEOffloadHost(unittest.TestCase):
    def test_private_plugin_alias_requires_identical_bytes(self):
        with tempfile.TemporaryDirectory(prefix="qmoe_plugin_alias_") as temporary:
            root = Path(temporary)
            requested = root / "libonnxruntime_providers_cuda.so"
            bundled = root / "onnxruntime" / "capi" / requested.name
            bundled.parent.mkdir(parents=True)
            requested.write_bytes(b"private-plugin")
            bundled.write_bytes(b"private-plugin")
            self.assertEqual(select_private_plugin_library(root, requested), bundled.resolve())
            bundled.write_bytes(b"stale-plugin")
            with self.assertRaisesRegex(RuntimeError, "different CUDA plugin"):
                select_private_plugin_library(root, requested)

    def test_actual_cpp_wave_planner(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="qmoe_offload_host_") as temporary:
            executable = Path(temporary) / "wave_plan"
            subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-fsanitize=undefined,address",
                    "-fno-pie",
                    "-no-pie",
                    "-I",
                    str(root / "onnxruntime"),
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(executable),
                ],
                input=HOST_TEST_SOURCE,
                text=True,
                check=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_identical_prepacked_model_payloads(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        resident = make_model("QMoE", banks, k=10, omit_prepacked=True)
        offload = make_model("QMoEOffload", banks, k=10, omit_prepacked=True)
        self.assertEqual(len(resident.graph.node[0].input), 21)
        self.assertEqual(list(resident.graph.node[0].input), list(offload.graph.node[0].input))
        self.assertNotIn("weights_prepacked", [attribute.name for attribute in offload.graph.node[0].attribute])
        self.assertEqual(
            [tensor.SerializeToString() for tensor in resident.graph.initializer],
            [tensor.SerializeToString() for tensor in offload.graph.initializer],
        )

    def test_e512_bank_exceeds_tiny_working_and_device_limits(self):
        banks = make_banks()
        total = sum(bank.nbytes for bank in banks.values())
        self.assertEqual(total, 512 * 55296)
        self.assertGreater(total, 16 << 20)
        model = make_model("QMoEOffload", banks, k=10, expert_wave_size=3, token_tile_size=2)
        self.assertEqual(model.graph.node[0].op_type, "QMoEOffload")

    def test_native_schema_slots_when_private_runtime_supplied(self):
        runtime = os.environ.get("ORT_QMOE_OFFLOAD_RUNTIME")
        if not runtime:
            self.skipTest("Private linked core not supplied; schema translation unit is separately compiled.")
        load_private_runtime(runtime)
        schemas = importlib.import_module("onnxruntime.capi._pybind_state").get_all_operator_schema()
        resident = next(schema for schema in schemas if schema.name == "QMoE" and schema.domain == "com.microsoft")
        offload = next(
            schema for schema in schemas if schema.name == "QMoEOffload" and schema.domain == "com.microsoft"
        )
        self.assertEqual(offload.since_version, 1)
        self.assertEqual([item.name for item in resident.inputs], [item.name for item in offload.inputs[:21]])
        self.assertEqual([item.name for item in resident.outputs], [item.name for item in offload.outputs[:1]])
        self.assertEqual(
            [item.name for item in offload.inputs[21:]],
            ["expert_blocks", "slab", "slot_tables", "fc2_bias", "token_phase"],
        )
        self.assertTrue(set(resident.attributes).issubset(offload.attributes))


def make_banks(experts=512, hidden=256, intermediate=128, biases=False):
    """Sample bytes directly in SM80 packed coordinates, not canonical raw INT4 coordinates."""
    generator = numpy.random.default_rng(1743)
    banks = {
        "W1": generator.integers(0, 256, (experts, hidden, intermediate), dtype=numpy.uint8),
        "W2": generator.integers(0, 256, (experts, intermediate, hidden // 2), dtype=numpy.uint8),
        "S1": generator.uniform(0.01, 0.03, (experts, 2 * intermediate, hidden // 32)).astype(numpy.float16),
        "S2": generator.uniform(0.01, 0.03, (experts, hidden, intermediate // 32)).astype(numpy.float16),
    }
    if biases:
        banks["B1"] = generator.normal(0, 0.07, (experts, 2 * intermediate)).astype(numpy.float16)
        banks["B2"] = generator.normal(0, 0.07, (experts, hidden)).astype(numpy.float16)
    return banks


def make_model(
    operator,
    banks,
    k=10,
    normalize=1,
    rank=2,
    extra_inputs=None,
    missing_scale=False,
    omit_prepacked=False,
    **overrides,
):
    experts, hidden, _ = banks["W1"].shape
    slots = ["X", "R", "W1", "S1", "", "W2", "S2"] + [""] * 14
    if "B1" in banks:
        slots[4] = "B1"
    if "B2" in banks:
        slots[7] = "B2"
    if missing_scale:
        slots[3] = ""
    initializers = [
        numpy_helper.from_array(bank, name=name) for name, bank in banks.items() if name != "S1" or not missing_scale
    ]
    if extra_inputs:
        for index, bank in extra_inputs.items():
            slots[index] = f"unsupported_{index}"
            initializers.append(numpy_helper.from_array(bank, name=slots[index]))
    attributes = {
        "k": k,
        "activation_type": "swiglu",
        "swiglu_fusion": 1,
        "normalize_routing_weights": normalize,
        "expert_weight_bits": 4,
        "block_size": 32,
        "quant_type": "int",
        "weights_prepacked": 1,
    }
    if operator == "QMoEOffload":
        attributes.update(
            packing_layout="cutlass_sm80_int4_v1",
            token_tile_size=2,
            expert_wave_size=1,
            expert_cache_bytes=0,
            gpu_working_bytes=8 << 20,
            pinned_staging_bytes=8 << 20,
        )
    attributes.update(overrides)
    if omit_prepacked:
        attributes.pop("weights_prepacked", None)
    shape = ["tokens", hidden] if rank == 2 else ["batch", "sequence", hidden]
    node = helper.make_node(operator, slots, ["Y"], name=operator, domain="com.microsoft", **attributes)
    graph = helper.make_graph(
        [node],
        f"test_{operator}",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["tokens", experts]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shape)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def load_private_runtime(runtime):
    root = Path(runtime).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("onnxruntime")
    native = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")
    if not Path(module.__file__).resolve().is_relative_to(root) or not Path(native.__file__).resolve().is_relative_to(
        root
    ):
        raise RuntimeError("Tests require the private core and Python extension, not an installed onnxruntime.")
    return module


def make_shared_model(blocks, bias, hidden, inter, k=3, **attributes):
    slots = ["X", "R"] + [""] * 19 + ["blocks", "", "", "bias", "phase"]
    options = dict(
        k=k,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
        expert_weight_bits=4,
        block_size=32,
        weights_prepacked=1,
        layout_version=1,
        hidden_size=hidden,
        inter_size=inter,
        layer_index=0,
        token_tile_size=2,
        gpu_working_bytes=2 << 20,
        pinned_staging_bytes=2 << 20,
    )
    options.update(attributes)
    experts = blocks.shape[0]
    graph = helper.make_graph(
        [helper.make_node("QMoEOffload", slots, ["Y", "rows", "calls", "counts"], domain="com.microsoft", **options)],
        "shared_offload",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["tokens", hidden]),
            helper.make_tensor_value_info("R", TensorProto.FLOAT16, ["tokens", experts]),
            helper.make_tensor_value_info("phase", TensorProto.INT8, ["tokens"]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["tokens", hidden]),
            helper.make_tensor_value_info("rows", TensorProto.INT32, [5, experts]),
            helper.make_tensor_value_info("calls", TensorProto.INT32, [5, experts]),
            helper.make_tensor_value_info("counts", TensorProto.INT32, [6]),
        ],
        initializer=[numpy_helper.from_array(blocks, "blocks"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


class TestQMoESharedGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GPU_CONFIG.gpu_tests:
            raise unittest.SkipTest("Shared GPU execution requires --gpu-tests.")
        cls.module, cls.device = register_private_plugin(GPU_CONFIG.runtime, GPU_CONFIG.plugin, GPU_CONFIG.device_index)
        sys.path.insert(0, str(Path(__file__).parent / "transformers"))
        cls.reference = importlib.import_module("qmoe_shared_layout_ref")
        cls.layout_fixtures = importlib.import_module("test_qmoe_shared_layout")

    def fixtures(self):
        generator = numpy.random.default_rng(987)
        experts = [self.layout_fixtures.random_expert(generator, 128, 64) for _ in range(16)]
        blocks = numpy.stack([self.reference.assemble_block(128, 64, **expert) for expert in experts])
        bias = numpy.stack([expert["fc2_bias"] for expert in experts])
        return generator, experts, blocks, bias

    def test_multi_tile_cpu_contributions_counts_and_empty_input(self):
        generator, experts, blocks, bias = self.fixtures()
        with tempfile.TemporaryDirectory(prefix="qmoe_shared_plugin_") as temporary:
            session = make_session(
                self.module,
                self.device,
                make_shared_model(blocks, bias, 128, 64),
                GPU_CONFIG.device_limit_bytes,
                Path(temporary) / "shared",
            )
            for tokens in (1, 9, 0):
                inputs = generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16)
                router = numpy.tile(numpy.arange(16, dtype=numpy.float16), (tokens, 1))
                phases = (numpy.arange(tokens) % 5).astype(numpy.int8)
                actual, route_rows, route_calls, counts = session.run(None, {"X": inputs, "R": router, "phase": phases})
                weights = numpy.exp(numpy.array([15, 14, 13], dtype=numpy.float64) - 15)
                weights /= weights.sum()
                expected = numpy.zeros((tokens, 128), dtype=numpy.float64)
                for rank, expert_id in enumerate((15, 14, 13)):
                    expert = dict(experts[expert_id], fc2_bias=numpy.zeros(128, dtype=numpy.float16))
                    contribution = self.reference.reference_expert(inputs.astype(numpy.float32), **expert).astype(
                        numpy.float16
                    )
                    expected += weights[rank] * (
                        contribution.astype(numpy.float64) + bias[expert_id].astype(numpy.float64)
                    )
                numpy.testing.assert_allclose(actual, expected.astype(numpy.float16), rtol=2e-3, atol=3e-4)
                numpy.testing.assert_array_equal(route_rows.sum(axis=1), 3 * counts[:5])
                numpy.testing.assert_array_equal(route_calls, (route_rows > 0).astype(numpy.int32))
                self.assertEqual(int(counts[:5].sum()), tokens)
                self.assertEqual(int(counts[5]), 0)
                self.assertTrue(numpy.isfinite(actual).all())
            assert_cuda_profile(session.end_profiling(), "QMoEOffload")

    def test_cpu_fp16_boundaries_match_cpu_reference(self):
        generator, experts, blocks, bias = self.fixtures()
        names = (
            "ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES",
            "ORT_QMOE_SHARED_RESIDENT_SLOTS",
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_CPU_THREADS",
        )
        previous = {name: os.environ.get(name) for name in names}
        lib = self.layout_fixtures.load_lib()
        try:
            os.environ.update(
                ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES="1",
                ORT_QMOE_SHARED_RESIDENT_SLOTS="0",
                ORT_QMOE_SHARED_PINNED_REUSE="1",
                ORT_QMOE_SHARED_CPU_THREADS="8",
            )
            for asynchronous in ("0", "1"):
                os.environ["ORT_QMOE_SHARED_ASYNC"] = asynchronous
                with tempfile.TemporaryDirectory(prefix="qmoe_cpu_fp16_") as temporary:
                    session = make_session(
                        self.module,
                        self.device,
                        make_shared_model(blocks, bias, 128, 64),
                        GPU_CONFIG.device_limit_bytes,
                        Path(temporary) / "shared",
                    )
                    for tokens in (1, 2, 9):
                        inputs = generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16)
                        router = numpy.tile(numpy.arange(16, dtype=numpy.float16), (tokens, 1))
                        phases = numpy.full(tokens, 2, dtype=numpy.int8)
                        actual, rows, calls, counts = session.run(None, {"X": inputs, "R": router, "phase": phases})
                        weights = numpy.exp(numpy.array([0, -1, -2], dtype=numpy.float64))
                        weights /= weights.sum()
                        expected = numpy.zeros((tokens, 128), dtype=numpy.float64)
                        natural = inputs.astype(numpy.float32)
                        for rank, expert_id in enumerate((15, 14, 13)):
                            contribution = numpy.empty((tokens, 128), dtype=numpy.float32)
                            for begin in range(0, tokens, 8):
                                self.assertEqual(
                                    lib.qmoe_shared_expert_contribution_fp16(
                                        blocks[expert_id].ctypes.data,
                                        128,
                                        64,
                                        natural[begin:].ctypes.data,
                                        min(8, tokens - begin),
                                        contribution[begin:].ctypes.data,
                                        1.0,
                                        0.0,
                                        float("inf"),
                                        0,
                                    ),
                                    1,
                                )
                            expected += weights[rank] * (
                                contribution.astype(numpy.float16).astype(numpy.float64)
                                + experts[expert_id]["fc2_bias"].astype(numpy.float64)
                            )
                        numpy.testing.assert_allclose(actual, expected.astype(numpy.float16), rtol=2e-3, atol=3e-4)
                        numpy.testing.assert_array_equal(rows.sum(axis=1), 3 * counts[:5])
                        numpy.testing.assert_array_equal(calls, (rows > 0).astype(numpy.int32))
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_grouped_expert_dispatch_is_byte_identical(self):
        generator, _, blocks, bias = self.fixtures()
        inputs = generator.normal(0, 0.4, (11, 128)).astype(numpy.float16)
        router = generator.normal(0, 1.0, (11, 16)).astype(numpy.float16)
        phases = (numpy.arange(11) % 5).astype(numpy.int8)
        outputs = {}
        previous = os.environ.get("ORT_QMOE_SHARED_GROUPED_CALLS")
        try:
            with tempfile.TemporaryDirectory(prefix="qmoe_shared_grouped_") as temporary:
                session = make_session(
                    self.module,
                    self.device,
                    make_shared_model(blocks, bias, 128, 64),
                    GPU_CONFIG.device_limit_bytes,
                    Path(temporary) / "shared",
                )
                for grouped in ("1", "3", "64"):
                    os.environ["ORT_QMOE_SHARED_GROUPED_CALLS"] = grouped
                    outputs[grouped] = session.run(None, {"X": inputs, "R": router, "phase": phases})
        finally:
            if previous is None:
                os.environ.pop("ORT_QMOE_SHARED_GROUPED_CALLS", None)
            else:
                os.environ["ORT_QMOE_SHARED_GROUPED_CALLS"] = previous
        for grouped in ("3", "64"):
            for expected, actual in zip(outputs["1"], outputs[grouped], strict=False):
                numpy.testing.assert_array_equal(actual, expected)

    def test_async_stream_callback_is_byte_identical(self):
        generator, _, blocks, bias = self.fixtures()
        names = (
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_GROUPED_CALLS",
            "ORT_QMOE_SHARED_CPU_THREADS",
            "ORT_QMOE_SHARED_POOL_TASK_BATCH",
        )
        previous = {name: os.environ.get(name) for name in names}
        feeds = []
        for tokens in (1, 2, 3, 4, 11, 1):
            feeds.append(
                {
                    "X": generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16),
                    "R": generator.normal(0, 1.0, (tokens, 16)).astype(numpy.float16),
                    "phase": (numpy.arange(tokens) % 5).astype(numpy.int8),
                }
            )
        results = {}
        try:
            for mode in ("sync", "async", "async_batch4"):
                os.environ["ORT_QMOE_SHARED_PINNED_REUSE"] = "1"
                os.environ["ORT_QMOE_SHARED_GROUPED_CALLS"] = "10"
                os.environ["ORT_QMOE_SHARED_CPU_THREADS"] = "8"
                os.environ["ORT_QMOE_SHARED_ASYNC"] = "0" if mode == "sync" else "1"
                os.environ["ORT_QMOE_SHARED_POOL_TASK_BATCH"] = "4" if mode == "async_batch4" else "1"
                with tempfile.TemporaryDirectory(prefix="qmoe_shared_async_") as temporary:
                    session = make_session(
                        self.module,
                        self.device,
                        make_shared_model(blocks, bias, 128, 64, token_tile_size=4),
                        GPU_CONFIG.device_limit_bytes,
                        Path(temporary) / "shared",
                    )
                    results[mode] = [session.run(None, feed) for _ in range(5) for feed in feeds]
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for arm in ("async", "async_batch4"):
            self.assertEqual(len(results["sync"]), len(results[arm]))
            for expected, actual in zip(results["sync"], results[arm], strict=False):
                for expected_output, actual_output in zip(expected, actual, strict=False):
                    numpy.testing.assert_array_equal(actual_output, expected_output)

    def test_async_resident_hits_match_cpu_contributions(self):
        generator, _, blocks, bias = self.fixtures()
        names = (
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_GROUPED_CALLS",
            "ORT_QMOE_SHARED_CPU_THREADS",
            "ORT_QMOE_SHARED_RESIDENT_SLOTS",
            "ORT_QMOE_SHARED_AUX_STREAM",
            "ORT_QMOE_SHARED_POOL_TASK_BATCH",
        )
        previous = {name: os.environ.get(name) for name in names}
        feeds = []
        for tokens in (1, 2, 3, 4, 2, 1):
            feeds.append(
                {
                    "X": generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16),
                    "R": generator.normal(0, 1.0, (tokens, 16)).astype(numpy.float16),
                    "phase": (numpy.arange(tokens) % 5).astype(numpy.int8),
                }
            )
        results = {}
        try:
            for arm in ("cpu", "single_stream", "aux_stream", "single_stream_batch4", "aux_stream_batch4"):
                os.environ.update(
                    ORT_QMOE_SHARED_PINNED_REUSE="1",
                    ORT_QMOE_SHARED_GROUPED_CALLS="10",
                    ORT_QMOE_SHARED_CPU_THREADS="8",
                )
                os.environ["ORT_QMOE_SHARED_ASYNC"] = "0" if arm == "cpu" else "1"
                os.environ["ORT_QMOE_SHARED_RESIDENT_SLOTS"] = "0" if arm == "cpu" else "6"
                os.environ["ORT_QMOE_SHARED_AUX_STREAM"] = "1" if arm.startswith("aux_stream") else "0"
                os.environ["ORT_QMOE_SHARED_POOL_TASK_BATCH"] = "4" if arm.endswith("_batch4") else "1"
                with tempfile.TemporaryDirectory(prefix="qmoe_shared_async_resident_") as temporary:
                    session = make_session(
                        self.module,
                        self.device,
                        make_shared_model(blocks, bias, 128, 64, token_tile_size=4),
                        GPU_CONFIG.device_limit_bytes,
                        Path(temporary) / "shared",
                    )
                    results[arm] = [session.run(None, feed) for _ in range(6) for feed in feeds]
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for arm in ("single_stream", "aux_stream", "single_stream_batch4", "aux_stream_batch4"):
            for run, (expected, actual) in enumerate(zip(results["cpu"], results[arm], strict=False)):
                numpy.testing.assert_allclose(actual[0], expected[0], rtol=3e-3, atol=1e-3, err_msg=f"{arm} run={run}")
                for statistic in range(1, 4):
                    numpy.testing.assert_array_equal(actual[statistic], expected[statistic])
        for arm in ("single_stream", "aux_stream"):
            for expected, actual in zip(results[arm], results[f"{arm}_batch4"], strict=False):
                for expected_output, actual_output in zip(expected, actual, strict=False):
                    numpy.testing.assert_array_equal(actual_output, expected_output)

    def test_phase_scores_and_margin_adapt_to_decode_without_prefill_pollution(self):
        generator, _, blocks, bias = self.fixtures()
        names = (
            "ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES",
            "ORT_QMOE_SHARED_RESIDENT_SLOTS",
            "ORT_QMOE_SHARED_RESIDENT_LOG",
            "ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES",
            "ORT_QMOE_SHARED_RESIDENT_PROMOTIONS",
            "ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT",
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_CPU_THREADS",
            "ORT_QMOE_SHARED_AUX_STREAM",
        )
        previous = {name: os.environ.get(name) for name in names}
        feeds = []
        for phase, tokens, repeats in ((0, 9, 8), (2, 1, 80), (0, 9, 2), (2, 1, 4)):
            router = numpy.arange(16, dtype=numpy.float16)
            if phase:
                router = router[::-1].copy()
            feed = {
                "X": generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16),
                "R": numpy.tile(router, (tokens, 1)),
                "phase": numpy.full(tokens, phase, dtype=numpy.int8),
            }
            feeds.extend([feed] * repeats)
        results, hits = {}, {}
        try:
            os.environ.update(
                ORT_QMOE_SHARED_RESIDENT_SLOTS="3",
                ORT_QMOE_SHARED_RESIDENT_LOG="1",
                ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES="4096",
                ORT_QMOE_SHARED_RESIDENT_PROMOTIONS="2",
                ORT_QMOE_SHARED_PINNED_REUSE="1",
                ORT_QMOE_SHARED_CPU_THREADS="8",
            )
            for mode, asynchronous, auxiliary in (("sync", "0", "0"), ("async", "1", "0"), ("aux", "1", "1")):
                os.environ.update(ORT_QMOE_SHARED_ASYNC=asynchronous, ORT_QMOE_SHARED_AUX_STREAM=auxiliary)
                for arm, separated, margin in (("legacy", "0", "100"), ("phase", "1", "100"), ("guarded", "0", "200")):
                    os.environ.update(
                        ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES=separated, ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT=margin
                    )
                    with tempfile.TemporaryDirectory(prefix="qmoe_phase_scores_") as temporary:
                        log_path = Path(temporary) / "native.log"
                        saved_stderr = os.dup(2)
                        try:
                            with log_path.open("wb") as log:
                                os.dup2(log.fileno(), 2)
                                session = make_session(
                                    self.module,
                                    self.device,
                                    make_shared_model(blocks, bias, 128, 64),
                                    GPU_CONFIG.device_limit_bytes,
                                    Path(temporary) / "shared",
                                )
                                results[mode, arm] = [session.run(None, feed) for feed in feeds]
                                del session
                                gc.collect()
                        finally:
                            os.dup2(saved_stderr, 2)
                            os.close(saved_stderr)
                        records = [
                            dict(field.split("=", 1) for field in line.split()[2:])
                            for line in log_path.read_text().splitlines()
                            if line.startswith("QMoEOffload resident_phase ")
                        ]
                        self.assertEqual(len(records), 5)
                        decode = next(record for record in records if record["phase"] == "2")
                        self.assertEqual(int(decode["routes"]), 252)
                        hits[mode, arm] = int(decode["hit_routes"])
                self.assertGreater(hits[mode, "phase"], hits[mode, "legacy"])
                self.assertGreater(hits[mode, "legacy"], hits[mode, "guarded"])
                self.assertEqual(hits[mode, "guarded"], 0)
                self.assertGreaterEqual(hits[mode, "phase"], 230)
                for arm in ("phase", "guarded"):
                    for expected, actual in zip(results[mode, "legacy"], results[mode, arm], strict=False):
                        numpy.testing.assert_allclose(actual[0], expected[0], rtol=3e-3, atol=1e-3)
                        for statistic in range(1, 4):
                            numpy.testing.assert_array_equal(actual[statistic], expected[statistic])
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_resident_cache_matches_cpu_contributions(self):
        generator, _, blocks, bias = self.fixtures()
        inputs = generator.normal(0, 0.4, (11, 128)).astype(numpy.float16)
        router = generator.normal(0, 1.0, (11, 16)).astype(numpy.float16)
        phases = (numpy.arange(11) % 5).astype(numpy.int8)
        feeds = {"X": inputs, "R": router, "phase": phases}
        previous = os.environ.get("ORT_QMOE_SHARED_RESIDENT_SLOTS")
        results = {}
        try:
            for slots in ("0", "4", "16"):
                os.environ["ORT_QMOE_SHARED_RESIDENT_SLOTS"] = slots
                with tempfile.TemporaryDirectory(prefix="qmoe_shared_resident_") as temporary:
                    session = make_session(
                        self.module,
                        self.device,
                        make_shared_model(blocks, bias, 128, 64),
                        GPU_CONFIG.device_limit_bytes,
                        Path(temporary) / "shared",
                    )
                    results[slots] = [session.run(None, feeds) for _ in range(6)]
        finally:
            if previous is None:
                os.environ.pop("ORT_QMOE_SHARED_RESIDENT_SLOTS", None)
            else:
                os.environ["ORT_QMOE_SHARED_RESIDENT_SLOTS"] = previous
        for slots in ("4", "16"):
            for run, (expected, actual) in enumerate(zip(results["0"], results[slots], strict=False)):
                row_error = numpy.abs(actual[0].astype(numpy.float32) - expected[0].astype(numpy.float32)).max(axis=1)
                row_relative = numpy.linalg.norm(
                    actual[0].astype(numpy.float32) - expected[0].astype(numpy.float32), axis=1
                ) / (numpy.linalg.norm(expected[0].astype(numpy.float32), axis=1) + 1e-9)
                numpy.testing.assert_allclose(
                    actual[0],
                    expected[0],
                    rtol=3e-3,
                    atol=1e-3,
                    err_msg=f"slots={slots} run={run} row_max_error={numpy.round(row_error, 5)} "
                    f"row_relative_l2={numpy.round(row_relative, 4)}",
                )
                for statistic in range(1, 4):
                    numpy.testing.assert_array_equal(actual[statistic], expected[statistic])

    def test_prefill_streaming_matches_cpu_contributions(self):
        generator, _, blocks, bias = self.fixtures()
        names = (
            "ORT_QMOE_SHARED_RESIDENT_SLOTS",
            "ORT_QMOE_SHARED_PIN_EXPERTS",
            "ORT_QMOE_SHARED_STREAM_ROWS",
            "ORT_QMOE_SHARED_STREAM_MIN_ROWS",
            "ORT_QMOE_SHARED_STREAM_WAVE",
            "ORT_QMOE_SHARED_STREAM_PROMOTIONS",
            "ORT_QMOE_SHARED_RESIDENT_LOG",
            "ORT_QMOE_SHARED_ASYNC",
            "ORT_QMOE_SHARED_PINNED_REUSE",
            "ORT_QMOE_SHARED_CPU_THREADS",
            "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE",
        )
        previous = {name: os.environ.get(name) for name in names}
        # The first tile streams exactly experts 13..15, so both landing buffers contain admitted candidates.
        feeds = [
            {
                "X": generator.normal(0, 0.4, (11, 128)).astype(numpy.float16),
                "R": numpy.tile(numpy.arange(16, dtype=numpy.float16), (11, 1)),
                "phase": numpy.zeros(11, dtype=numpy.int8),
            }
        ]
        for tokens in (11, 1, 19, 2, 11):
            feeds.append(
                {
                    "X": generator.normal(0, 0.4, (tokens, 128)).astype(numpy.float16),
                    "R": generator.normal(0, 1.0, (tokens, 16)).astype(numpy.float16),
                    "phase": (numpy.arange(tokens) % 5).astype(numpy.int8),
                }
            )
        results, records = {}, {}
        try:
            for arm in ("cpu", "stream", "stream_reuse", "stream_all", "stream_all_reuse"):
                for name in names:
                    os.environ.pop(name, None)
                if arm != "cpu":
                    os.environ.update(
                        ORT_QMOE_SHARED_RESIDENT_SLOTS="4",
                        ORT_QMOE_SHARED_PIN_EXPERTS="1",
                        ORT_QMOE_SHARED_STREAM_ROWS="8",
                        ORT_QMOE_SHARED_STREAM_WAVE="2",
                        ORT_QMOE_SHARED_STREAM_PROMOTIONS="2",
                        ORT_QMOE_SHARED_RESIDENT_LOG="1",
                        ORT_QMOE_SHARED_ASYNC="1",
                        ORT_QMOE_SHARED_PINNED_REUSE="1",
                        ORT_QMOE_SHARED_CPU_THREADS="8",
                        ORT_QMOE_SHARED_STREAM_MIN_ROWS="1" if arm.startswith("stream_all") else "2",
                        ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE="1" if arm.endswith("_reuse") else "0",
                    )
                with tempfile.TemporaryDirectory(prefix="qmoe_prefill_stream_") as temporary:
                    log_path = Path(temporary) / "native.log"
                    saved_stderr = os.dup(2)
                    try:
                        with log_path.open("wb") as log:
                            os.dup2(log.fileno(), 2)
                            session = make_session(
                                self.module,
                                self.device,
                                make_shared_model(blocks, bias, 128, 64),
                                GPU_CONFIG.device_limit_bytes,
                                Path(temporary) / "shared",
                            )
                            results[arm] = [session.run(None, feed) for _ in range(4) for feed in feeds]
                            del session
                            gc.collect()
                    finally:
                        os.dup2(saved_stderr, 2)
                        os.close(saved_stderr)
                    records[arm] = [
                        dict(field.split("=", 1) for field in line.split()[2:])
                        for line in log_path.read_text().splitlines()
                        if line.startswith("QMoEOffload stream ")
                    ]
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.assertEqual(records["cpu"], [])
        for arm in ("stream", "stream_reuse", "stream_all", "stream_all_reuse"):
            self.assertEqual(len(records[arm]), 1)
            record = records[arm][0]
            self.assertGreater(int(record["tiles"]), 0)
            self.assertGreater(int(record["waves"]), int(record["tiles"]))
            self.assertGreater(int(record["routes"]), 0)
            if arm.startswith("stream_all"):
                self.assertEqual(int(record["cpu_routes"]), 0)
            else:
                self.assertGreater(int(record["cpu_routes"]), 0)
            if arm.endswith("_reuse"):
                self.assertGreater(int(record["reused_promotions"]), 0)
            else:
                self.assertEqual(int(record["reused_promotions"]), 0)
            for run, (expected, actual) in enumerate(zip(results["cpu"], results[arm], strict=False)):
                numpy.testing.assert_allclose(actual[0], expected[0], rtol=3e-3, atol=1e-3, err_msg=f"{arm} run={run}")
                for statistic in range(1, 4):
                    numpy.testing.assert_array_equal(actual[statistic], expected[statistic])
        for arm in ("stream", "stream_all"):
            for expected, actual in zip(results[arm], results[f"{arm}_reuse"], strict=False):
                for expected_output, actual_output in zip(expected, actual, strict=False):
                    numpy.testing.assert_array_equal(actual_output, expected_output)

    def test_invalid_shared_contract_and_phase(self):
        _, _, blocks, bias = self.fixtures()
        for attributes in (
            {"layout_version": 2},
            {"weights_prepacked": -1},
            {"block_size": 64},
            {"layer_index": -1},
            {"hidden_size": 127},
            {"gpu_working_bytes": 1},
        ):
            with self.subTest(attributes=attributes):
                with self.assertRaises(Exception):
                    session = make_session(
                        self.module,
                        self.device,
                        make_shared_model(blocks, bias, 128, 64, **attributes),
                        GPU_CONFIG.device_limit_bytes,
                    )
                    session.run(
                        None,
                        {
                            "X": numpy.zeros((1, 128), dtype=numpy.float16),
                            "R": numpy.zeros((1, 16), dtype=numpy.float16),
                            "phase": numpy.zeros(1, dtype=numpy.int8),
                        },
                    )
        session = make_session(
            self.module, self.device, make_shared_model(blocks, bias, 128, 64), GPU_CONFIG.device_limit_bytes
        )
        for phase in (-1, 5):
            with self.assertRaisesRegex(Exception, "token_phase"):
                session.run(
                    None,
                    {
                        "X": numpy.zeros((1, 128), dtype=numpy.float16),
                        "R": numpy.zeros((1, 16), dtype=numpy.float16),
                        "phase": numpy.array([phase], dtype=numpy.int8),
                    },
                )


class TestHyperConnectionGateActivationGPU(unittest.TestCase):
    """Fused gate_activation must match the separate Div/Sigmoid/Mul nodes bitwise."""

    @classmethod
    def setUpClass(cls):
        if not GPU_CONFIG.gpu_tests:
            raise unittest.SkipTest("GPU execution requires explicit parent-coordinated --gpu-tests.")
        cls.module, cls.device = register_private_plugin(GPU_CONFIG.runtime, GPU_CONFIG.plugin, GPU_CONFIG.device_index)

    @staticmethod
    def build(proto_type, fused):
        rows, branches, hidden = 5, 4, 96
        constants = {"four": numpy.array(4.0), "two": numpy.array(2.0)}
        dtype = {TensorProto.FLOAT16: numpy.float16, TensorProto.FLOAT: numpy.float32}[proto_type]
        initializers = [numpy_helper.from_array(value.astype(dtype), name=name) for name, value in constants.items()]
        ms = {"domain": "com.microsoft"}
        nodes = []
        if fused:
            nodes += [
                helper.make_node(
                    "HyperConnectionPostMix",
                    ["streams", "block", "post_gate"],
                    ["post"],
                    gate_activation=1,
                    gate_input_scale=0.25,
                    gate_output_scale=2.0,
                    **ms,
                ),
                helper.make_node("HyperConnectionPreMix", ["streams", "pre_gate"], ["pre"], gate_activation=1, **ms),
                helper.make_node("GatedAdd", ["x", "y", "add_gate"], ["added"], gate_activation=1, **ms),
            ]
        else:
            nodes += [
                helper.make_node("Div", ["post_gate", "four"], ["post_div"]),
                helper.make_node("Sigmoid", ["post_div"], ["post_sig"]),
                helper.make_node("Mul", ["post_sig", "two"], ["post_weight"]),
                helper.make_node("HyperConnectionPostMix", ["streams", "block", "post_weight"], ["post"], **ms),
                helper.make_node("Sigmoid", ["pre_gate"], ["pre_weight"]),
                helper.make_node("HyperConnectionPreMix", ["streams", "pre_weight"], ["pre"], **ms),
                helper.make_node("Sigmoid", ["add_gate"], ["add_weight"]),
                helper.make_node("GatedAdd", ["x", "y", "add_weight"], ["added"], **ms),
            ]
        value = helper.make_tensor_value_info
        graph = helper.make_graph(
            nodes,
            "gate_activation",
            [
                value("streams", proto_type, [rows, branches, hidden]),
                value("block", proto_type, [rows, hidden]),
                value("post_gate", proto_type, [rows, branches, 1]),
                value("pre_gate", proto_type, [rows, branches]),
                value("x", proto_type, [rows, hidden]),
                value("y", proto_type, [rows, hidden]),
                value("add_gate", proto_type, [rows, 1]),
            ],
            [value("post", proto_type, None), value("pre", proto_type, None), value("added", proto_type, None)],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        return model, (rows, branches, hidden), dtype

    def check(self, proto_type):
        fused, (rows, branches, hidden), dtype = self.build(proto_type, True)
        reference, _, _ = self.build(proto_type, False)
        generator = numpy.random.default_rng(31)
        feed = {
            "streams": generator.normal(0, 1, (rows, branches, hidden)),
            "block": generator.normal(0, 1, (rows, hidden)),
            "post_gate": generator.normal(0, 6, (rows, branches, 1)),
            "pre_gate": generator.normal(0, 6, (rows, branches)),
            "x": generator.normal(0, 1, (rows, hidden)),
            "y": generator.normal(0, 1, (rows, hidden)),
            "add_gate": generator.normal(0, 6, (rows, 1)),
        }
        feed = {name: value.astype(dtype) for name, value in feed.items()}
        outputs = []
        for model in (fused, reference):
            session = make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)
            outputs.append(session.run(None, feed))
        for name, actual, expected in zip(("post", "pre", "added"), *outputs, strict=False):
            numpy.testing.assert_array_equal(actual, expected, err_msg=name)

    def test_float16_matches_separate_nodes(self):
        self.check(TensorProto.FLOAT16)

    def test_float32_matches_separate_nodes(self):
        self.check(TensorProto.FLOAT)


class TestScaledSiLUGatedGPU(unittest.TestCase):
    """ScaledSiLU(gated=1) must match Split + QuickGelu + Mul (ORT's fused form of Sigmoid + Mul) bitwise."""

    @classmethod
    def setUpClass(cls):
        if not GPU_CONFIG.gpu_tests:
            raise unittest.SkipTest("GPU execution requires explicit parent-coordinated --gpu-tests.")
        cls.module, cls.device = register_private_plugin(GPU_CONFIG.runtime, GPU_CONFIG.plugin, GPU_CONFIG.device_index)

    @staticmethod
    def build(proto_type, fused, shape):
        value = helper.make_tensor_value_info
        if fused:
            nodes = [helper.make_node("ScaledSiLU", ["x"], ["y"], gated=1, domain="com.microsoft")]
        else:
            nodes = [
                helper.make_node("Split", ["x"], ["gate", "up"], axis=-1, num_outputs=2),
                helper.make_node("QuickGelu", ["gate"], ["silu"], alpha=1.0, domain="com.microsoft"),
                helper.make_node("Mul", ["silu", "up"], ["y"]),
            ]
        graph = helper.make_graph(
            nodes, "gated_silu", [value("x", proto_type, list(shape))], [value("y", proto_type, None)]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        return model

    def check(self, proto_type):
        dtype = {TensorProto.FLOAT16: numpy.float16, TensorProto.FLOAT: numpy.float32}[proto_type]
        for shape in ((1, 1280), (2, 1280), (3, 2, 96), (7, 6)):
            x = numpy.random.default_rng(17).normal(0, 4, shape).astype(dtype)
            outputs = []
            for fused in (True, False):
                session = make_session(
                    self.module, self.device, self.build(proto_type, fused, shape), GPU_CONFIG.device_limit_bytes
                )
                outputs.append(session.run(None, {"x": x})[0])
            self.assertEqual(outputs[0].shape, shape[:-1] + (shape[-1] // 2,))
            numpy.testing.assert_array_equal(outputs[0], outputs[1], err_msg=str(shape))

    def test_float16_matches_separate_nodes(self):
        self.check(TensorProto.FLOAT16)

    def test_float32_matches_separate_nodes(self):
        self.check(TensorProto.FLOAT)


def select_private_plugin_library(runtime, plugin):
    library = Path(plugin).resolve(strict=True)
    bundled = Path(runtime).resolve() / "onnxruntime" / "capi" / library.name
    if bundled.exists() and bundled.resolve() != library:
        if (
            hashlib.file_digest(bundled.open("rb"), "sha256").digest()
            != hashlib.file_digest(library.open("rb"), "sha256").digest()
        ):
            raise RuntimeError("Private package contains a different CUDA plugin than the requested library.")
        library = bundled.resolve()
    return library


def register_private_plugin(runtime, plugin, device_index=0):
    library = select_private_plugin_library(runtime, plugin)
    module = load_private_runtime(runtime)
    if "CUDAExecutionProvider" in module.get_available_providers():
        try:
            module.unregister_execution_provider_library("CUDAExecutionProvider")
        except Exception as error:
            if "was not registered" not in str(error):
                raise
    module.register_execution_provider_library("CUDAExecutionProvider", str(library))
    devices = [
        device
        for device in module.get_ep_devices()
        if device.ep_name == "CUDAExecutionProvider" and device.device.type == module.OrtHardwareDeviceType.GPU
    ]
    if device_index < 0 or device_index >= len(devices):
        raise RuntimeError("Private CUDA plugin did not expose the requested device; fallback is not allowed.")
    if str(library) not in Path("/proc/self/maps").read_text():
        raise RuntimeError("The explicitly selected private CUDA plugin is not mapped into this process.")
    print(json.dumps({"private_runtime": str(Path(module.__file__).resolve()), "private_plugin": str(library)}))
    return module, devices[device_index]


def make_session(module, device, model, limit, profile_prefix=None, graphs=False, cpu_initializers=None):
    options = module.SessionOptions()
    options.graph_optimization_level = module.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    options.add_session_config_entry("ep.cuda.gpu_device_mem_limit", str(limit))
    options.add_session_config_entry("ep.cuda.enable_cuda_graph", "1" if graphs else "0")
    if cpu_initializers:
        for name, value in cpu_initializers.items():
            options.add_initializer(name, value)
    options.add_provider_for_devices(
        [device],
        {
            "gpu_device_mem_limit": str(limit),
            "gpu_mem_limit": str(limit),
            "enable_cuda_graph": "1" if graphs else "0",
        },
    )
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    session = module.InferenceSession(model.SerializeToString(), sess_options=options)
    session.disable_fallback()
    return session


def assert_cuda_profile(profile, operator):
    events = json.loads(Path(profile).read_text())
    nodes = [event for event in events if event.get("args", {}).get("op_name") == operator]
    if not nodes or any(event["args"].get("provider") != "CUDAExecutionProvider" for event in nodes):
        raise AssertionError(f"Expected {operator} to execute on the explicitly loaded CUDA plugin, not a fallback.")


class TestQMoEOffloadGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GPU_CONFIG.gpu_tests:
            raise unittest.SkipTest("GPU execution requires explicit parent-coordinated --gpu-tests.")
        cls.module, cls.device = register_private_plugin(GPU_CONFIG.runtime, GPU_CONFIG.plugin, GPU_CONFIG.device_index)

    def compare(self, banks, k, shape, tied=False, normalize=1, wave=1, omit_prepacked=False):
        rows = int(numpy.prod(shape[:-1]))
        generator = numpy.random.default_rng(9721)
        inputs = generator.normal(0, 0.5, shape).astype(numpy.float16)
        router = (
            numpy.zeros((rows, banks["W1"].shape[0]), dtype=numpy.float16)
            if tied
            else generator.normal(0, 1.5, (rows, banks["W1"].shape[0])).astype(numpy.float16)
        )
        feeds = {"X": inputs, "R": router}
        with tempfile.TemporaryDirectory(prefix="qmoe_offload_parity_") as temporary:
            resident = make_session(
                self.module,
                self.device,
                make_model("QMoE", banks, k=k, normalize=normalize, rank=len(shape), omit_prepacked=omit_prepacked),
                GPU_CONFIG.device_limit_bytes,
                Path(temporary) / "resident",
            )
            offload = make_session(
                self.module,
                self.device,
                make_model(
                    "QMoEOffload",
                    banks,
                    k=k,
                    normalize=normalize,
                    rank=len(shape),
                    expert_wave_size=wave,
                    omit_prepacked=omit_prepacked,
                ),
                GPU_CONFIG.device_limit_bytes,
                Path(temporary) / "offload",
            )
            expected = resident.run(None, feeds)[0]
            for _ in range(3):
                actual = offload.run(None, feeds)[0]
                self.assertEqual(actual.dtype, numpy.float16)
                self.assertTrue(numpy.isfinite(actual).all())
                numpy.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-4)
            assert_cuda_profile(resident.end_profiling(), "QMoE")
            assert_cuda_profile(offload.end_profiling(), "QMoEOffload")
            difference = numpy.abs(actual.astype(numpy.float32) - expected.astype(numpy.float32))
            print(
                json.dumps(
                    {
                        "experts": banks["W1"].shape[0],
                        "k": k,
                        "shape": shape,
                        "tied": tied,
                        "normalize": normalize,
                        "wave": wave,
                        "max_abs_error": float(difference.max()),
                        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    }
                )
            )
            del resident, offload
            gc.collect()

    def test_decode_prefill_and_rank_three(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        for shape in ((1, 128), (7, 128), (2, 3, 128)):
            for normalize in (0, 1):
                for tied in (False, True):
                    with self.subTest(shape=shape, normalize=normalize, tied=tied):
                        self.compare(banks, 3, shape, tied=tied, normalize=normalize, wave=1)

    def test_e512_top10_zero_cache_union_exceeds_waves(self):
        banks = make_banks()
        for rows in (1, 7):
            for normalize in (0, 1):
                for wave in (1, 3):
                    with self.subTest(rows=rows, normalize=normalize, wave=wave):
                        self.compare(banks, 10, (rows, 256), normalize=normalize, wave=wave)
        self.compare(banks, 10, (7, 256), tied=True, normalize=1, wave=3)

    def test_model_projection_geometry_and_omitted_prepack(self):
        banks = make_banks(experts=16, hidden=2560, intermediate=640, biases=True)
        for rows in (1, 7):
            for normalize in (0, 1):
                with self.subTest(rows=rows, normalize=normalize):
                    self.compare(banks, 10, (rows, 2560), normalize=normalize, omit_prepacked=True)

    def test_nonzero_fc1_fc2_biases_keep_original_combine_precision(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64, biases=True)
        for rows in (1, 7):
            for normalize in (0, 1):
                for wave in (1, 2):
                    with self.subTest(rows=rows, normalize=normalize, wave=wave):
                        self.compare(banks, 3, (rows, 128), normalize=normalize, wave=wave)

    def test_single_route_with_grouped_row_padding(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        for rows in (1, 7):
            self.compare(banks, 1, (rows, 128))

    def test_fused_fp32_decode_contract_is_rejected(self):
        banks = make_banks(experts=16, hidden=2560, intermediate=640)
        with self.assertRaisesRegex(Exception, "QMoEOffload.*fused FP32 GEMV"):
            make_session(self.module, self.device, make_model("QMoEOffload", banks, k=2), GPU_CONFIG.device_limit_bytes)

    def test_malformed_attributes(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        cases = [
            {"expert_weight_bits": 8},
            {"fc1_expert_weight_bits": 8},
            {"block_size": 64},
            {"quant_type": "fp4"},
            {"weights_prepacked": 0},
            {"packing_layout": "unknown"},
            {"activation_type": "relu"},
            {"swiglu_fusion": 2},
            {"use_sparse_mixer": 1},
            {"normalize_routing_weights": 2},
            {"accuracy_level": 4},
            {"expert_cache_bytes": 1},
            {"expert_wave_size": 0},
            {"expert_wave_size": 32},
            {"token_tile_size": 0},
            {"token_tile_size": 1 << 62},
            {"gpu_working_bytes": 1},
            {"pinned_staging_bytes": 1},
            {"k": 0},
            {"k": 33},
        ]
        for attributes in cases:
            with self.subTest(attributes=attributes):
                model = make_model("QMoEOffload", banks, **attributes)
                with self.assertRaisesRegex(Exception, "QMoEOffload|[Oo]verflow|SafeInt|swiglu_fusion|Sparse mixer"):
                    make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)

    def test_malformed_input_contracts(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        for slot, value in (
            (4, numpy.zeros((32, 127), dtype=numpy.float16)),
            (7, numpy.zeros((32, 127), dtype=numpy.float16)),
            (8, banks["W1"]),
            (11, numpy.zeros((32, 128, 2), dtype=numpy.uint8)),
            (14, numpy.zeros((1, 32), dtype=numpy.float16)),
        ):
            with self.subTest(slot=slot):
                model = make_model("QMoEOffload", banks, extra_inputs={slot: value})
                with self.assertRaisesRegex(Exception, "QMoEOffload.*(unsupported|does not support|bias inputs)"):
                    make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)
        with self.assertRaisesRegex(Exception, "QMoEOffload.*(immutable|missing)"):
            make_session(
                self.module,
                self.device,
                make_model("QMoEOffload", banks, missing_scale=True),
                GPU_CONFIG.device_limit_bytes,
            )
        invalid_banks = dict(banks)
        invalid_banks["S1"] = banks["S1"][:, :, :-1].copy()
        with self.assertRaisesRegex(Exception, "QMoEOffload.*scales"):
            make_session(
                self.module, self.device, make_model("QMoEOffload", invalid_banks), GPU_CONFIG.device_limit_bytes
            )
        model = make_model("QMoEOffload", banks)
        initializer = next(tensor for tensor in model.graph.initializer if tensor.name == "W1")
        model.graph.initializer.remove(initializer)
        model.graph.input.append(helper.make_tensor_value_info("W1", TensorProto.UINT8, banks["W1"].shape))
        with self.assertRaisesRegex(Exception, "QMoEOffload.*immutable"):
            make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)
        model = make_model("QMoEOffload", banks)
        for value in (*model.graph.input, *model.graph.output):
            value.type.tensor_type.elem_type = TensorProto.FLOAT
        with self.assertRaisesRegex(Exception, "(QMoEOffload|[Ii]nvalid|[Tt]ype)"):
            make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)
        model = make_model("QMoEOffload", banks)
        scale = next(tensor for tensor in model.graph.initializer if tensor.name == "S1")
        scale.CopyFrom(numpy_helper.from_array(banks["S1"].astype(numpy.float32), name="S1"))
        with self.assertRaisesRegex(Exception, "(QMoEOffload|[Ii]nvalid|[Tt]ype)"):
            make_session(self.module, self.device, model, GPU_CONFIG.device_limit_bytes)

    def test_borrowed_unaligned_cpu_scale_is_rejected(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        storage = bytearray(banks["S1"].nbytes + 1)
        scale = numpy.ndarray(banks["S1"].shape, dtype=numpy.float16, buffer=storage, offset=1)
        scale[...] = banks["S1"]
        value = self.module.OrtValue.ortvalue_from_numpy(scale)
        self.assertEqual(value.data_ptr() % 2, 1)
        with self.assertRaisesRegex(Exception, "QMoEOffload.*aligned storage"):
            make_session(
                self.module,
                self.device,
                make_model("QMoEOffload", banks),
                GPU_CONFIG.device_limit_bytes,
                cpu_initializers={"S1": value},
            )

    def test_empty_input_and_wrong_router_shape(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        session = make_session(
            self.module, self.device, make_model("QMoEOffload", banks), GPU_CONFIG.device_limit_bytes
        )
        output = session.run(
            None, {"X": numpy.empty((0, 128), numpy.float16), "R": numpy.empty((0, 32), numpy.float16)}
        )[0]
        self.assertEqual(output.shape, (0, 128))
        with self.assertRaisesRegex(Exception, "QMoEOffload.*router shape"):
            session.run(None, {"X": numpy.zeros((2, 128), numpy.float16), "R": numpy.zeros((1, 32), numpy.float16)})

    def test_graph_capture_is_rejected(self):
        banks = make_banks(experts=32, hidden=128, intermediate=64)
        with self.assertRaisesRegex(Exception, "(QMoEOffload.*(graph|eager)|CUDA.*graph|graph.*CPU)"):
            session = make_session(
                self.module, self.device, make_model("QMoEOffload", banks), GPU_CONFIG.device_limit_bytes, graphs=True
            )
            binding = session.io_binding()
            inputs = self.module.OrtValue.ortvalue_from_numpy(numpy.ones((1, 128), numpy.float16), "cuda", 0)
            router = self.module.OrtValue.ortvalue_from_numpy(numpy.zeros((1, 32), numpy.float16), "cuda", 0)
            output = self.module.OrtValue.ortvalue_from_numpy(numpy.empty((1, 128), numpy.float16), "cuda", 0)
            binding.bind_ortvalue_input("X", inputs)
            binding.bind_ortvalue_input("R", router)
            binding.bind_ortvalue_output("Y", output)
            for _ in range(5):
                session.run_with_iobinding(binding)

    def test_fresh_process_tiny_device_limit(self):
        for arm in ("offload", "resident"):
            with self.subTest(arm=arm):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--gpu-tests",
                        "--runtime",
                        GPU_CONFIG.runtime,
                        "--plugin",
                        GPU_CONFIG.plugin,
                        "--device-index",
                        str(GPU_CONFIG.device_index),
                        "--memory-arm",
                        arm,
                    ],
                    check=True,
                )


def run_memory_arm(args):
    module, device = register_private_plugin(args.runtime, args.plugin, args.device_index)
    banks = make_banks()
    bank_bytes = sum(bank.nbytes for bank in banks.values())
    limit = 16 << 20
    if bank_bytes <= limit:
        raise AssertionError("Constrained-memory fixture must be larger than the aggregate physical device limit.")
    operator = "QMoEOffload" if args.memory_arm == "offload" else "QMoE"
    model = make_model(operator, banks, k=10)
    if args.memory_arm == "resident":
        try:
            make_session(module, device, model, limit)
        except Exception as error:
            if not any(word in str(error).lower() for word in ("gpu_device_mem_limit", "budget", "device limit")):
                raise
            print(
                json.dumps(
                    {
                        "arm": "resident",
                        "bank_bytes": bank_bytes,
                        "device_limit_bytes": limit,
                        "expected_budget_rejection": str(error),
                    }
                )
            )
            return 0
        raise AssertionError(
            "Resident bank unexpectedly fit below its own byte extent; physical budget was not enforced."
        )
    with tempfile.TemporaryDirectory(prefix="qmoe_offload_bound_") as temporary:
        for trial in range(3):
            session = make_session(module, device, model, limit, Path(temporary) / f"offload_{trial}")
            for rows in (1, 7, 1, 7):
                output = session.run(
                    None, {"X": numpy.ones((rows, 256), numpy.float16), "R": numpy.zeros((rows, 512), numpy.float16)}
                )[0]
                if not numpy.isfinite(output).all():
                    raise AssertionError("Constrained offload output is not finite.")
            assert_cuda_profile(session.end_profiling(), "QMoEOffload")
            del session
            gc.collect()
    print(
        json.dumps(
            {
                "arm": "offload",
                "bank_bytes": bank_bytes,
                "device_limit_bytes": limit,
                "gpu_working_bytes": 8 << 20,
                "expert_cache_bytes": 0,
                "trials": 3,
            }
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host-only", action="store_true", help="No CUDA registration or execution.")
    mode.add_argument("--gpu-tests", action="store_true", help="Run only after parent GPU/MPS coordination.")
    parser.add_argument("--runtime", help="Private built Python package directory, never installedlibs.")
    parser.add_argument("--plugin", help="Explicit private libonnxruntime_providers_cuda.so.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-limit-bytes", type=int, default=256 << 20)
    parser.add_argument(
        "--shared-only", action="store_true", help="Run shared-block tests instead of legacy GPU cases."
    )
    parser.add_argument("--memory-arm", choices=("offload", "resident"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runtime:
        os.environ["ORT_QMOE_OFFLOAD_RUNTIME"] = args.runtime
    if args.gpu_tests and (not args.runtime or not args.plugin):
        parser.error(
            "--gpu-tests requires both --runtime and --plugin; installed libraries and fallback are not accepted."
        )
    if args.memory_arm:
        if not args.gpu_tests:
            parser.error("--memory-arm requires explicit --gpu-tests.")
        return run_memory_arm(args)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestQMoEOffloadHost)
    if args.gpu_tests:
        vars(GPU_CONFIG).update(vars(args))
        if not args.shared_only:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestQMoEOffloadGPU))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestQMoESharedGPU))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestHyperConnectionGateActivationGPU))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestScaledSiLUGatedGPU))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
