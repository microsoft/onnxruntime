import sys
import argparse

from .model import ModelImp, SupportedModels

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t",
                        "--model_type",
                        required=False,
                        type=str,
                        default="onnx",
                        choices=SupportedModels,
                        help="Supported model types in the list: " + SupportedModels)
    
    parser.add_argument("-p",
                        "--model_path",
                        required=True,
                        type=str,
                        help="Path of the model") # might need to pass model dir instead of the exact file path 
                                                  # in case of other model types

    parser.add_argument("-g",
                        "--use_gpu",
                        required=False,
                        action="store_true",
                        help="Run on cuda device")


    return parser.parse_args()

def start_processing(inputFilePath: str, outputFilePath: str, model: ModelImp):
    try:
        with open(inputFilePath, 'r', encoding = 'utf-8') as inFileHandle:
            with open(outputFilePath, 'w', encoding = 'utf-8') as outFileHandle:
                line = inFileHandle.readline()
                line = line.strip()
                while(line):
                    result = model.Eval(line, outFileHandle)
                    outFileHandle.write(result)
                    outFileHandle.write("\n")
                    line = inFileHandle.readline()
                    line = line.strip()
    except:
        raise

def initilize_processing(inputFilePath: str, outputFilePath: str):
    try:
        args = parse_arguments()
        model = ModelImp(args)
        start_processing(inputFilePath, outputFilePath, model)
    except Exception as e:
        sys.stderr.write(str(e))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file> <output_file>")
        sys.exit(1)
    
    initilize_processing(sys.argv[1], sys.argv[2])

