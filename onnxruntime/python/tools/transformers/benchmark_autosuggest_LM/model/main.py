import sys
import argparse
import time

import myutils
from model import ModelImp, SupportedModels

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t",
                        "--model_type",
                        required=False,
                        type=str,
                        default="onnx",
                        choices=SupportedModels,
                        help="Supported model types in the list: " + str(SupportedModels))
    
    parser.add_argument("-m",
                        "--model_path",
                        required=True,
                        type=str,
                        help="Path of the model") # might need to pass model dir instead of the exact file path 
                                                  # in case of other model types

    parser.add_argument("-i",
                        "--input_file",
                        required=True,
                        default=None,
                        help="Input file for test data")

    parser.add_argument("-o",
                        "--output_file",
                        required=True,
                        default=None,
                        help="Output file for test data")

    parser.add_argument("-n",
                        "--no_of_suggestions",
                        required=False,
                        default=8,
                        help="Number of suggestions for output file")

    parser.add_argument("-g",
                        "--use_gpu",
                        required=False,
                        action="store_true",
                        help="Run on cuda device")

    parser.add_argument("-b",
                        "--beam_size",
                        required=False,
                        default=2,
                        help="Beam size for the model")
    
    parser.add_argument("--tokenizer_path",
                        required=False,
                        default="model_files/",
                        help="Path where tokenizer files reside")

    return parser.parse_args()

def start_processing(inputFilePath: str, outputFilePath: str, model: ModelImp):
    try:
        count = 0
        with open(inputFilePath, 'r', encoding = 'utf-8') as inFileHandle:
            with open(outputFilePath, 'w', encoding = 'utf-8') as outFileHandle:
                '''
                outFileHandle.write("Iterations\ten+de\tInferTime\tSearchTime\te2e\tResult\n")
                line = inFileHandle.readline()
                line = line.strip()
                while(line):
                
                    result = model.Eval(line, outFileHandle)
                    outFileHandle.write(result)
                    outFileHandle.write("\n")
                    line = inFileHandle.readline()
                    line = line.strip()
                '''
                myutils.outFileHandler(outFileHandle)
                line = inFileHandle.readline()
                total_time = 0.0

                while(line):
                    myutils.counterset = False

                    start = time.perf_counter()
                    result = model.Eval(line)
                    end = time.perf_counter()

                    time_taken = (end - start) * 1000
                    total_time += time_taken
                    count += 1
                    if result == "[]" and myutils.counterset == False:
                        outFileHandle.write(str(0) + "\t")
                        outFileHandle.write(str(0) + "\t")
                        outFileHandle.write(str(0) + "\t")
                    outFileHandle.write(result + "\t")
                    outFileHandle.write(str(time_taken) + "\n")
                    
                    line = inFileHandle.readline()
        
                print("Total Queries: " + str(count))

            print("Average latency: ", str(total_time / count))
            print("Mask any counter:" + str(myutils.mask_any_counter))
    except:
        raise

def initilize_processing():
    try:
        args = parse_arguments()
        model = ModelImp(args)
        start_processing(args.input_file, args.output_file, model)
    except Exception as e:
        sys.stderr.write(str(e))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python main.py -m <model_path> -i <input_file> -o <output_file>")
        sys.exit(1)
    
    initilize_processing()