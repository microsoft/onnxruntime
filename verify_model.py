import onnx

model_path = "model_1.onnx"
m = onnx.load(model_path)

print("IR version:", m.ir_version)
print("Opset imports:", [(d.domain or "ai.onnx", d.version) for d in m.opset_import])

print("\nNodes:")
for i, n in enumerate(m.graph.node):
    print(f"{i:3d}: name={n.name!r}, op_type={n.op_type!r}, domain={n.domain!r}")
