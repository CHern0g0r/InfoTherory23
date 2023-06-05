from torch.nn import Module


class Coder(Module):
    def __init__(self,
                 coder_class,
                 *args,
                 weights=None,
                 **kwargs):
        super().__init__()
        self.coder = coder_class(*args)
        if weights:
            self.coder.load_state_dict(
                torch.load(weights)
            )

    def forward(self, X):
        out = self.coder(X)
        return out


class Encoder(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X


class Decoder(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        return X


class Pipeline(Module):

    @staticmethod
    def create(classes):
        pass

    def __init__(self, encoder, decoder, quant):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.q = quant

    def forward(self, X):
        return X
