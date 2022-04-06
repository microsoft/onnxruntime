from onnxflow import TrainingGraph, Graph
import onnxflow
import onnx

class MyGraph(TrainingGraph):
    def __init__(self, base_model):
        super(MyGraph, self).__init__()
        self.loss = onnxflow.loss.MSELoss()
        self.base_model = base_model

    def build(self):
        outputs = self.base_model.graph.output
        lossful_graph = self.loss(self.base_model, outputs[0].name)
        return lossful_graph

onnxfile = 'models/simple_model.onnx'
model = onnx.load(onnxfile)

graph = MyGraph(model)

# remove in case of any model other than simple_model.onnx
graph.requires_grad('_original_module.fc1.weight')
graph.requires_grad('_original_module.fc1.bias')
graph.requires_grad('_original_module.fc2.weight')
graph.requires_grad('_original_module.fc2.bias')

gradient_graph = graph()

parameters = graph.parameters()
onnxflow.save(parameters, 'parameters.of')

optimizer = onnxflow.optim.AdamW()
optimizer_graph = optimizer(gradient_graph)

onnx.save(gradient_graph, "gradient_graph.onnx")
onnx.save(optimizer_graph, "optimizer_graph.onnx")
