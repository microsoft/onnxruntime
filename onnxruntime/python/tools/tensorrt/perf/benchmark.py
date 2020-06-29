import csv
import timeit
from datetime import datetime
import numpy
import logging

from BERTSquad import *
from Resnet50 import *

logger = logging.getLogger('')

MODELS = {
    "bert-squad": (BERTSquad, "bert-squad"),
    "resnet50": (Resnet50, "resnet50")
}

def get_latency_result(runtimes, batch_size):
    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
    latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    return {
        "test_times": len(runtimes),
        "latency_variance": "{:.2f}".format(latency_variance),
        "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000.0),
        "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000.0),
        "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000.0),
        "average_latency_ms": "{:.2f}".format(latency_ms),
        "QPS": "{:.2f}".format(throughput),
    }

def inference_ort(ort_session, result_template, repeat_times, batch_size):
    result = {}
    runtimes = timeit.repeat(lambda: ort_session.inference(), number=1, repeat=repeat_times)
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(runtimes, batch_size))
    return result

def run_onnxruntime(models=MODELS):
    import onnxruntime

    for ep in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
        if (ep not in onnxruntime.get_available_providers()):
            logger.error(
                "No {} support".format(ep)
            )
            continue

        for name in models.keys():
            info = models[name] 
            model = info[0]
            path = info[1]
            fp16 = False
            num_inputs = 2
            batch_size = 1
            sequence_length = 1
            optimize_onnx = False

            pwd = os.getcwd()
            if not os.path.exists(path):
                os.mkdir(path)
            os.chdir(path)

            # create onnxruntime inference session
            model = model()
            sess = model.get_session()
            sess.set_providers([ep])

            result_template = {
                "engine": "onnxruntime",
                "version": onnxruntime.__version__,
                "device": ep,
                "optimizer": optimize_onnx,
                "fp16": fp16,
                "io_binding": False,
                "model_name": model.get_model_name(),
                "inputs": num_inputs,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "datetime": str(datetime.now()),
            }

            result = inference_ort(model, result_template, 1, 1)
            print(result)
            model.postprocess()

            os.chdir(pwd)

run_onnxruntime()
