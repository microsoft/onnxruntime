import onnxruntime 
import numpy as np

HAILO_ONNX_MODEL = "yolox_tiny_leaky_hailo.onnx"

def main():
    data = np.random.randn(1, 3, 416, 416).astype('f')
    
    ep_list = ['HailoExecutionProvider']
    session = onnxruntime.InferenceSession(HAILO_ONNX_MODEL, providers=ep_list)
    outputs = session.run([output.name for output in session.get_outputs()], {session.get_inputs()[0].name: data})


if __name__ == "__main__":
    main()