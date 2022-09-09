import torch.nn as nn


class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()

    def forward(self, params):
        raise NotImplementedError("FW in MetaModel should be implemented in subclass.")
