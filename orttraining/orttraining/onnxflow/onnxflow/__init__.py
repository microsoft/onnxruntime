from .graph import Graph, TrainingGraph
from .loss import MSELoss, CrossEntropyLoss
from .optim import AdamW

def save(parameters, path_to_file):
    with open(path_to_file, 'wb') as file_object:
        file_object.write(parameters.SerializeToString())
