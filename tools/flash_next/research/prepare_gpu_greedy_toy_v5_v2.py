from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "gpu_greedy_toy_v5.py").read_text()
    source = replace_once(source, "argmax/toy-v5", "argmax/toy-v5-v2")
    source = replace_once(source, 'type="gpt2"', 'type="llama"')
    source = replace_once(source, "    import onnxruntime_genai as og",
                          "    import faulthandler\n    faulthandler.enable()\n    import onnxruntime_genai as og")
    source = replace_once(source, "        model = og.Model(str(folder))",
        '        print(json.dumps(dict(stage="model", model=folder.name)), flush=True)\n'
        "        model = og.Model(str(folder))")
    source = replace_once(source, "            gen = og.Generator(model, params)",
        '            print(json.dumps(dict(stage="generator", case=case)), flush=True)\n'
        "            gen = og.Generator(model, params)")
    with (ROOT / "gpu_greedy_toy_v5_v2.py").open("x") as stream:
        stream.write(source)
    source = (ROOT / "run_gpu_greedy_toy_v5_probe2.py").read_text()
    source = source.replace("toy-probe2", "toy-probe3")
    source = source.replace("gpu_greedy_toy_v5.py", "gpu_greedy_toy_v5_v2.py")
    source = source.replace("/toy-v5/float32-32", "/toy-v5-v2/float32-32")
    with (ROOT / "run_gpu_greedy_toy_v5_probe3.py").open("x") as stream:
        stream.write(source)
