import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, hidden_node=3):
        self.input_node = 1
        self.hidden_node = hidden_node
        self.output_node = 1
        self.w1 = np.random.rand(self.hidden_node, self.input_node)
        self.b1 = np.random.rand(self.hidden_node, 1)
        self.w2 = np.random.rand(self.output_node, self.hidden_node)
        self.b2 = np.random.rand(self.output_node, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

