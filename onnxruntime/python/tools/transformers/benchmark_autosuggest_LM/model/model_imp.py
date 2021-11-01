import json
import sys
import time

from wrapper.onnx import load_onnx
from model_runner import ModelRunner
from tokenizer import Tokenizer
from wrapper.onnx import OnnxModelWrapper

PADTOKENIDs = {
    "onnx" : 0
}

class ModelImp:
    def  __init__(self,  args = None):
        self._args = args
        self._is_onnx_model = args.model_type == "onnx"
        self._device = args.device
        self._pad_token_id = PADTOKENIDs[args.model_type]

        if self._is_onnx_model:
            print("Loading the following model:" + args.model_path)
            start_time = time.perf_counter()
            self.onnx_model = load_onnx(args.model_path, device=args.device)
            end_time = time.perf_counter()
            print("Time taken for loading onnx model:" + str(end_time - start_time))
            io_binding = True if 'cuda' in args.device.lower() else False
            self.wrapped_model = OnnxModelWrapper(
                self.onnx_model,
                self._pad_token_id,
                max_batch_size=1,
                io_binding=io_binding)
        else:
            print("Currently only onnx is supported")
            sys.exit(0)
      
        self._tokenizer = Tokenizer(args.tokenizer_path)
        self._modelrunner = ModelRunner()
        self._num_suggestions = args.num_suggestions  

    def Eval(self, data):
        top = self._num_suggestions

        outputs, probs = self._modelrunner.autocomplete(
            self._args,
            self.wrapped_model,
            self._tokenizer,
            data,
            pad_token_id=self._pad_token_id,
            is_onnx_model=self._is_onnx_model)

        outputs = outputs[0]
        probs = probs[0]
        outputs = [output.strip() for output in outputs]
        if not data.endswith(' '):
            prefix = data.split(' ')[-1]
            results = [ answer for answer in outputs if answer.lower() != prefix.lower() and len(answer) > 1][:top]
        else:
            results = [ answer for answer in outputs if len(answer) > 1][:top]
        results = json.dumps(results)
        return results

