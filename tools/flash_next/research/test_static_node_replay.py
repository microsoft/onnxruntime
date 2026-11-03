from pathlib import Path
import re
import unittest

from build_static_node_replay import transform


class StaticNodeReplayTests(unittest.TestCase):
    def originals(self):
        repository = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")
        originals = {}
        for name in ("contrib_ops/webgpu/bert/linear_attention_gates.h",
                     "contrib_ops/webgpu/bert/linear_attention_gates.cc",
                     "contrib_ops/webgpu/bert/scaled_silu.h"):
            originals[name] = (repository / "onnxruntime" / name).read_text()
        originals["contrib_ops/webgpu/bert/scaled_silu.cc"] = (
            repository / "build-int4-split-silu-v3/scaled_silu.cc").read_text()
        for name in ("quickgelu_mul_overlay.h", "quickgelu_mul_overlay.cc"):
            originals[name] = (repository.parent / "qwen38-int4-quickgelu-next/build" / name).read_text()
        originals["core/providers/webgpu/webgpu_context.cc"] = (
            repository.parent / "qwen38-int4-qmoe-node-replay/build-v5/overlay/core/providers/webgpu/webgpu_context.cc"
        ).read_text()
        for name in ("core/providers/webgpu/webgpu_context.h", "core/providers/webgpu/compute_context.h"):
            originals[name] = (
                repository.parent / "qwen38-int4-qmoe-node-replay/build-v5/overlay" / name).read_text()
        return originals

    def test_scoped_overlay(self):
        originals = self.originals()
        result = transform(dict(originals))
        self.assertEqual(set(result), set(originals))
        self.assertEqual(sum(text.count("return static_replay_.Run") for text in result.values()), 3)
        self.assertIn('static_replay_{"SplitScaledSiLUF16", {0}}', result["contrib_ops/webgpu/bert/scaled_silu.h"])
        split = result["contrib_ops/webgpu/bert/scaled_silu.cc"]
        self.assertLess(split.index('split[0] > 0'), split.index("return static_replay_.Run"))
        self.assertLess(split.index("context.Output(1"), split.index("return static_replay_.Run"))
        context = result["core/providers/webgpu/webgpu_context.cc"]
        self.assertIn('ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_STATIC_NODE_REPLAY", 0) == 1', context)
        self.assertEqual(context.count("!captured_node_uniform"), 1)
        name = "core/providers/webgpu/webgpu_context.h"
        for struct in ("CapturedCommandInfo", "CommandRecordingState"):
            pattern = rf"struct {struct} \{{.*?\n\}};"
            self.assertEqual(re.search(pattern, originals[name], re.S).group(),
                             re.search(pattern, result[name], re.S).group())
        self.assertIn("snapshot.buffers = bind_buffers", context)
        self.assertIn("snapshot.layout = bind_group_layout", context)

    def test_exact_matmul_guard_preserves_original_fallback(self):
        repository = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")
        originals = self.originals()
        originals["contrib_ops/webgpu/quantization/matmul_nbits.h"] = (
            repository / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.h").read_text()
        originals["contrib_ops/webgpu/quantization/matmul_nbits.cc"] = (
            repository / "build-int4-shared-post-v4/matmul_nbits.cc").read_text()
        result = transform(originals)
        source = result["contrib_ops/webgpu/quantization/matmul_nbits.cc"]
        for fragment in ('bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr',
                         '!context.EnableMatmulFp32Accumulation()',
                         'K_ == 2560 && (N_ == 512 || N_ == 640)', 'K_ == 10240 && N_ == 324',
                         'if (!exact_route) return compute();',
                         'return static_replay_.Run(context, {y}, compute);'):
            self.assertIn(fragment, source)
        self.assertEqual(source.count('return ApplyMatMulNBits(a, b, scales, zero_points, bias, K_, N_'), 1)

    def test_owned_bounded_per_kernel_cache(self):
        source = Path(__file__).with_name("static_node_replay.h").read_text()
        for fragment in ("first->Shape().Size() != (row_elements > 0 ? row_elements : first->Shape().GetDims().back())",
                         "context.NodeReplayProfilingEnabled()", "context.CanCaptureNodeCommands()",
                         "entries_.size() == kCapacity", "kCapacity = 64", "entries_.erase(oldest)",
                         "context.PrepareNodeBinding(entry.commands.front()", "entry.commands.size() == 1",
                         "gsl::finally", "input->SizeInBytes() > 64", "key.push_back(bytes[i])"):
            self.assertIn(fragment, source)
        self.assertNotIn("CreateGPUTensor", source)
        self.assertLess(source.index("context.FlushNodeCommands()"), source.index("entries_.push_back"))
        self.assertNotIn("static std::vector", source)
        self.assertNotIn("reinterpret_cast<uintptr_t>(tensor->DataRaw())", source)
        self.assertIn("entry.bindings.buffers == expected", source)
        self.assertIn("context.BeginNodeBindingCapture", source)
        self.assertIn("context.EndNodeBindingCapture", source)
        self.assertEqual(source.count("location.name == WEBGPU_BUFFER"), 2)
        self.assertEqual(source.count("location.mem_type == OrtMemType::OrtMemTypeDefault"), 2)
        self.assertIn("candidate.prototype_id == entry.id && candidate.buffers == buffers", source)
        self.assertIn("context.AppendNodeCommands(bound->commands)", source)
        self.assertIn("binding_entries_.size() == kCapacity", source)

    def test_hc_logical_row_preserves_validation_and_shaders(self):
        repository = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")
        originals = self.originals()
        header = "contrib_ops/webgpu/bert/hyper_connection_pre_mix.h"
        premix = "contrib_ops/webgpu/bert/hyper_connection_pre_mix.cc"
        post = "hc_post_norm_overlay.cc"
        originals[header] = (repository / "onnxruntime" / header).read_text()
        originals[premix] = (repository / "build-int4-hc-fusion/hyper_connection_pre_mix.cc").read_text()
        originals[post] = (repository.parent / "qwen38-int4-hc-post-norm/build-v8" / post).read_text()
        result = transform(dict(originals))
        for name, checks in ((premix, ("ResolveStreamShape", "ResolveGateShape", "context.Output")),
                             (post, ("ValidateReduced", "ValidateScale", "maxComputeWorkgroupStorageSize",
                                     "context.Output"))):
            source = result[name]
            for check in checks:
                self.assertLess(source.index(check), source.index("return static_replay_.Run"))
            self.assertIn("}, params.branches * params.hidden);", source)
            if name == premix:
                marker = "Status HyperConnectionPreMixProgram::GenerateShaderCode"
                end = "\nHyperConnectionPreMix::HyperConnectionPreMix"
            else:
                marker = "class HCPostNormProgram"
                end = "\nclass HCPostNormF16"
            self.assertEqual(source.split(marker, 1)[1].split(end, 1)[0],
                             originals[name].split(marker, 1)[1].split(end, 1)[0])
        self.assertIn('static_replay_{"HyperConnectionPreMix", {0, 1}}', result[header])
        self.assertIn('std::initializer_list<int>{0, 1, 2, 3, 4, 5}', result[post])


if __name__ == "__main__":
    unittest.main()
