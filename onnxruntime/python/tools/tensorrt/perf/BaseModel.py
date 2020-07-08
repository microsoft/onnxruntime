class BaseModel(object): 
    def __init__(self, model_name):
        self.model_name_ = model_name 
        self.session_ = None
        self.onnx_zoo_test_data_set_dir = None
        self.outputs_ = []

    def get_model_name(self):
        return self.model_name_

    def get_session(self):
        return self.session_

    def get_onnx_zoo_test_data_dir(self):
        return self.onnx_zoo_test_data_dir_

    def get_outputs(self):
        return self.outputs_
