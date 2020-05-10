import torch
from torch import nn
from Attention import Attention
from constants import DEVICE
from Encoder import Encoder
from Decoder import Decoder


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim,
                 value_size=128,
                 key_size=128,
                 isAttended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True, epoch=None):
        key, value, lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, lens, text_input, epoch=epoch)
        else:
            predictions = self.decoder(key, value, lens, text=None, isTrain=False, epoch=None)
        return predictions