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

    def save(self):



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
    def create(classes, args, weights=(None, None)):
        if not all(map(
            lambda x: len(x) == 2,
            [classes, args]
        )):
            assert False

        if len(weights) == 1:
            weights = (None, None)
        
        enc, dec = map(
            lambda x: x[0](x[1], x[2]),
            zip(classes, args, weights)
        )

    def __init__(self, encoder, decoder, quant):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.q = quant

    def forward(self, X):
        out = self.enc(X)
        out = self.q(X)
        out = self.dec(X)
        return X
