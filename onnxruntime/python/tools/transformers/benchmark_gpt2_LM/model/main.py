import sys
import argparse
import time
import myutils

from model_imp import ModelImp

SupportedModels = ['onnx', 'dlis']
MaxSuggestions = 8

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
                        type=str,
                        help="Input file for test data")

    parser.add_argument("-o",
                        "--output_file",
                        required=True,
                        type=str,
                        help="Output file for test data")

    parser.add_argument("-r",
                        "--ref_file",
                        required=False,
                        type=str,
                        help="Used for only calc_appg.py to know the predictions")

    parser.add_argument("-n",
                        "--num_suggestions",
                        required=False,
                        type=int,
                        default=MaxSuggestions,
                        help="Number of suggestions for output file")

    parser.add_argument("--num_beams",
                        required=False,
                        type=int,
                        default=0,
                        help="Number of beams in beam search")

    parser.add_argument("--device",
                        required=False,
                        type=str,
                        default='cuda:0')

    parser.add_argument("-g",
                        "--use_gpu",
                        required=False,
                        action="store_true",
                        help="Run on cuda device")

    parser.add_argument('--num_words',
                        type=int,
                        default=8,
                        help='Number next words to generate')

    parser.add_argument('--length_penalty',
                        type=int,
                        default=1.6,
                        help='Number next words to generate')
    
    parser.add_argument("--tokenizer_path",
                        required=False,
                        default="tokenizer_files/",
                        help="Path where tokenizer files reside")

    return parser.parse_args()

def start_processing(inputFilePath: str, outputFilePath: str, model: ModelImp):
    try:
        count = 0
        with open(inputFilePath, 'r', encoding = 'utf-8') as inFileHandle:
            with open(outputFilePath, 'w', encoding = 'utf-8') as outFileHandle:
                myutils.outFileHandler(outFileHandle)
                line = inFileHandle.readline()
                total_time = 0.0

                while(line):
                    start = time.perf_counter()
                    result = model.Eval(line)
                    end = time.perf_counter()
                    time_taken = (end - start) * 1000

                    count += 1
                    total_time += time_taken

                    if result == "[]":
                        print("No result found for the input:" + line)
                    
                    outFileHandle.write(result + "\t")
                    outFileHandle.write(str(time_taken) + "\n")
                    
                    line = inFileHandle.readline()
                print("Total Queries: " + str(count))

            print("Average latency: ", str(total_time / count))
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