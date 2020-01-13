from BertOnnxModel import BertOnnxModel

class BertOnnxModelTF(BertOnnxModel):
    def __init(self, model, num_heads, hidden_size, sequence_length, verbose):
        super().__init__(model, num_heads, hidden_size, sequence_length, verbose)