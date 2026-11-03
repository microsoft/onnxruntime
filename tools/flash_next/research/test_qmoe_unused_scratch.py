from pathlib import Path
import subprocess
import tempfile
import unittest

from build_qmoe_unused_scratch import FC1_CONDITION, FC2_CONDITION, SELECTOR, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1/qmoe.cc")


class UnusedScratchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text()
        cls.changed = transform(cls.original)

    def test_only_allocation_lifetime_and_private_selector_change(self):
        restored = self.changed.replace(SELECTOR, "", 1)
        for index, suffix, name, condition in (
                (2, "fc1", "fuse_fc1_activation", FC1_CONDITION),
                (4, "fc2", "fuse_fc2_final_mix", FC2_CONDITION)):
            restored = restored.replace(
                f"    const bool {name} = {condition};\n"
                f"    if (!omit_unused_scratch || !{name}) {{\n"
                f"      temporary({index}, dtype, {suffix}_output_shape);\n"
                "    }",
                f"    Tensor& {suffix}_outputs = temporary({index}, dtype, {suffix}_output_shape);", 1)
            restored = restored.replace(f"    if ({name}) {{", f"    if ({condition}) {{", 1)
            restored = restored.replace(
                f"    }} else {{\n    Tensor& {suffix}_outputs = temporary({index}, dtype, {suffix}_output_shape);\n",
                "    } else {\n", 1)
        self.assertEqual(restored, self.original)

    def test_private_selector_is_frozen_and_restricts_decode_without_replay(self):
        self.assertIn("static const int lazy_unused_scratch", SELECTOR)
        self.assertIn("value == 0 || value == 1", SELECTOR)
        self.assertIn("scratch == nullptr && moe_params.num_rows == 1", SELECTOR)
        self.assertIn("return value;", SELECTOR)

    def test_actual_cpp_allocation_snippets_cover_all_fusion_mode_combinations(self):
        start = self.changed.index("    std::array<std::optional<Tensor>, 5> local_scratch;")
        end = self.changed.index("    const uint32_t k =", start)
        temporary = self.changed[start:end]
        start = self.changed.index("    TensorShape fc1_output_shape({static_cast<int64_t>(routed_rows), fc1_output_size});")
        end = self.changed.index("    if (fuse_fc1_activation)", start)
        fc1 = self.changed[start:end]
        start = self.changed.index("    TensorShape fc2_output_shape({static_cast<int64_t>(routed_rows), N_fc2});")
        end = self.changed.index("    if (fuse_fc2_final_mix)", start)
        fc2 = self.changed[start:end]
        stub = r"""
#include <array>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>
#define ORT_ENFORCE(ok, message) do { if (!(ok)) std::abort(); } while (false)
int selected = 0;
bool first = false, second = false;
template<class T> T ParseEnvironmentVariableWithDefault(const char* name, T) {
    std::string key(name);
    if (key == "ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH") return selected;
    return key == "ORT_WEBGPU_QMOE_FC1_ACTIVATION" ? first : second;
}
using MLDataType = int;
struct TensorShape {
    std::vector<int64_t> dims;
    TensorShape(std::initializer_list<int64_t> values) : dims(values) {}
    bool operator==(const TensorShape&) const = default;
};
struct Tensor {
    int dtype;
    TensorShape shape;
    int DataType() const { return dtype; }
    const TensorShape& Shape() const { return shape; }
};
struct Context {
    std::vector<int> allocated;
    int requested_index = 0;
    Tensor CreateGPUTensor(int dtype, const TensorShape& shape) {
        allocated.push_back(requested_index);
        return {dtype, shape};
    }
};
bool CanFuseFc1Activation(Context&, bool enabled) { return enabled; }
bool CanFuseFc2FinalMix(Context&, bool enabled) { return enabled; }
void run(int rows, bool replay) {
    Context context;
    struct { int num_rows; int inter_size = 640; } moe_params{rows};
    std::array<std::optional<Tensor>, 5> saved;
    auto* scratch = replay ? &saved : nullptr;
    constexpr int dtype = 1, fc1_output_size = 1280, N_fc2 = 2560;
    const int routed_rows = rows * 10;
"""
        temporary = temporary.replace(
            "tensor.emplace(context.CreateGPUTensor(type, shape));",
            "context.requested_index = static_cast<int>(index);\n"
            "        tensor.emplace(context.CreateGPUTensor(type, shape));")
        body = (SELECTOR + temporary +
                "    temporary(0, dtype, TensorShape({512}));\n"
                "    temporary(1, dtype, TensorShape({10}));\n" + fc1 +
                "    (void)fc1_activated;\n"
                "    if (!fuse_fc1_activation) temporary(2, dtype, fc1_output_shape);\n" + fc2 +
                "    if (!fuse_fc2_final_mix) temporary(4, dtype, fc2_output_shape);\n")
        end = r"""
    std::vector<int> expected{0, 1};
    bool omit = selected == 1 && rows == 1 && !replay;
    if (!omit || !first) expected.push_back(2);
    expected.push_back(3);
    if (!omit || !second) expected.push_back(4);
    if (context.allocated != expected) std::abort();
}
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    selected = std::atoi(argv[1]);
    for (int rows : {1, 2}) for (bool replay : {false, true})
        for (bool f1 : {false, true}) for (bool f2 : {false, true}) {
            first = f1; second = f2; run(rows, replay);
        }
}
"""
        with tempfile.TemporaryDirectory(prefix="unused-scratch-test-", dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "scratch.cpp"
            binary = Path(tmp) / "scratch"
            source.write_text(stub + body + end)
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(binary)], check=True)
            for flag in (0, 1):
                subprocess.run([str(binary), str(flag)], check=True)


if __name__ == "__main__":
    unittest.main()
