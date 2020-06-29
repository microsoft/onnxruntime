class BaseModel(object): 
    def __init__(self, model_name):
        self.model_name_ = model_name 
        self.session_ = None

    def get_model_name(self):
        return self.model_name_

    def get_session(self):
        return self.session_
