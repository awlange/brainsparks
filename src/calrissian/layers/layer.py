
class Layer(object):

    def __init__(self, type):
        self.type = type
        self.input_shape = (0, 0)
        self.output_shape = (0, 0)
