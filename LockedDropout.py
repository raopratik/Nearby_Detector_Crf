# Reference: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html

from torch import nn

class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, train = True):
        x = x.clone()
        if train:
            mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
            mask = mask/(1-self.p)
            mask = mask.expand_as(x)
            x *= mask
        return x