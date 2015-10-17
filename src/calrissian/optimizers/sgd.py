from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent optimization
    """

    def __init__(self, alpha=0.01, n_epochs=1, mini_batch_size=10):
        """
        :param alpha: learning rate
        """
        super().__init__()
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

    def optimize(self, network, data_X, data_Y):
        """
        :return: optimized network
        """
        print("hello")
