from optimum.onnxruntime import ORTModelForCausalLM

name = "mistralai/Mistral-7B-Instruct-v0.1"
model = ORTModelForCausalLM.from_pretrained(
    name,
    export=True,
    use_auth_token=True,
)
model.save_pretrained(name.split("/")[-1] + "-onnx")
