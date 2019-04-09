import torch
import random
import numpy as np


class TorchConfig(object):

    def __init__(self, seed=2019):
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():  # USE_CUDA
            torch.cuda.manual_seed(self.seed)
