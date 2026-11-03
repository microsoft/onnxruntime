import json
from pathlib import Path
import subprocess

import onnx


def main():
    original = Path("/home/sayanshaw/repos/qwen38-int4-exact-expert")
    directory = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-node-replay/fixtures")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(directory)
    subprocess.run(["sudo", "-n", "ln", str(original / "fixtures/model.onnx.data"),
                    str(directory / "model.onnx.data")], check=True)
    fixtures = json.loads((original / "fixtures.json").read_text())
    for fixture in fixtures:
        model = onnx.load(original / "fixtures" / fixture["fixture"], load_external_data=False)
        tensors = {tensor.name: tensor for tensor in model.graph.initializer}
        assert not model.graph.input and tensors["x"].data_type == tensors["router"].data_type == onnx.TensorProto.FLOAT16
        for name in ("x", "router"):
            tensor = tensors[name]
            model.graph.input.append(onnx.helper.make_tensor_value_info(name, tensor.data_type, tensor.dims))
        keep = [tensor for tensor in model.graph.initializer if tensor.name not in ("x", "router")]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        onnx.save(model, directory / fixture["fixture"])
    (directory.parent / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    print(json.dumps(dict(fixtures=len(fixtures), weights_hardlinked=True, weights_copied=False)))


if __name__ == "__main__":
    main()
