from torch.nn import (
    MSELoss,
    Module
)


L1 = MSELoss


class L2(Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, X):
        return X.mean()
