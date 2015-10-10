
from src.sandbox.layer import Layer


class Input(Layer):

    def __init__(self, size=0):
        super().__init__()
        self.type = "Input"
        self.size = size
