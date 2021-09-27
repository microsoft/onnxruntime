from io import TextIOWrapper

SupportedModels = ["onnx", "dlis"]

class ModelImp:
    def __init__(self, modelType : str, modelPath: str):
        self.mType = modelType
        self.mPath = modelPath

    def Eval(self, line: str, outFileHandle: TextIOWrapper):



