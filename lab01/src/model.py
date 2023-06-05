from torch.nn import Module


class Encoder(Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class Decoder(Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class Pipeline(Module):
    def __init__(self, encoder, decoder, quant):
        self.enc = encoder
        self.dec = decoder
        self.q = quant

    def forward(self, X):
        return X
