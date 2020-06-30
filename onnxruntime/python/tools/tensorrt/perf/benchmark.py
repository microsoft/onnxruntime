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

    results = []
    for name in models.keys():
        info = models[name] 
        model = info[0]
        path = info[1]

        pwd = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)


        for ep in ["TensorrtExecutionProvider", "CUDAExecutionProvider"]:
            if (ep not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep))
                continue

            # these settings are temporary
            fp16 = False
            sequence_length = 1
            optimize_onnx = False
            repeat_times = 10
            batch_size = 1

            # create onnxruntime inference session
            print("Initializing {} with {}...".format(name, ep))
            model_obj = model()
            
            sess = model_obj.get_session()
            if ep == "CUDAExecutionProvider":
                sess.set_providers([ep])

            result_template = {
                "engine": "onnxruntime",
                "version": onnxruntime.__version__,
                "device": ep,
                "optimizer": optimize_onnx,
                "fp16": fp16,
                "io_binding": False,
                "model_name": model_obj.get_model_name(),
                "inputs": len(sess.get_inputs()),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "datetime": str(datetime.now()),
            }

            print(sess.get_providers())
            print("Inferencing {} with {} ...".format(model_obj.get_model_name(), ep))

            result = inference_ort(model_obj, result_template, repeat_times, batch_size)

            print(result)
            results.append(result)
            model_obj.postprocess()

        os.chdir(pwd)

    return results

def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "fp16", "optimizer", "io_binding", "model_name", "inputs", "batch_size",
            "sequence_length", "datetime", "test_times", "QPS", "average_latency_ms", "latency_variance",
            "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--detail_csv", required=False, default=None, help="CSV file for saving detail results.")

    # parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    results = run_onnxruntime()

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = args.detail_csv or f"benchmark_detail_{time_stamp}.csv"
    output_details(results, csv_filename)

    # csv_filename = args.result_csv or f"benchmark_summary_{time_stamp}.csv"
    # output_summary(results, csv_filename, args)


if __name__ == "__main__":
    main()
