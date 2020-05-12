import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pBLSTM import pBLSTM
from LockedDropout import LockedDropout
class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''

    def __init__(self, input_dim=40, hidden_dim=512, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                            bidirectional=True)
        self.pb1 = pBLSTM(input_dim=hidden_dim*4, hidden_dim=hidden_dim)
        self.pb2 = pBLSTM(input_dim=hidden_dim*4, hidden_dim=hidden_dim)
        self.pb3 = pBLSTM(input_dim=hidden_dim*4, hidden_dim=hidden_dim)
        self.lockedropout = LockedDropout()
        ### Add code to define the blocks of pBLSTMs! ###

        self.key_network = nn.Linear(hidden_dim * 2, value_size)
        self.value_network = nn.Linear(hidden_dim * 2, key_size)

    def forward(self, x, lens, isTrain):

        rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        packed_outputs, _ = self.lstm(rnn_inp)
        ### Use the outputs and pass it through the pBLSTM blocks! ###
        output_padded, output_lengths = pad_packed_sequence(packed_outputs)

        output_padded = self.lockedropout(output_padded, isTrain)
        output_padded, output_lengths = self.pb_block(output_padded,
                                                      output_lengths, self.pb1, isTrain)

        output_padded, output_lengths = self.pb_block(output_padded,
                                                      output_lengths, self.pb2, isTrain)

        output_padded, output_lengths = self.pb_block(output_padded,
                                                      output_lengths, self.pb3, isTrain)


        keys = self.key_network(output_padded)
        value = self.value_network(output_padded)
        return keys, value, output_lengths

    def pb_block(self, output_padded, output_lengths, pb, isTrain):
        output_padded, output_lengths = \
            self.half_network(output_padded=output_padded, output_lengths=output_lengths)
        output_padded, output_lengths = pb(output_padded, output_lengths)
        if pb != self.pb3:
            output_padded = self.lockedropout(output_padded, isTrain)
        return output_padded, output_lengths

    def half_network(self, output_padded, output_lengths):
        S, N, E = output_padded.shape

        if S % 2 == 1:
            output_padded = output_padded[:-1, :, :]

        output_lengths = [curr_len // 2 for curr_len in output_lengths]

        output_padded = torch.transpose(output_padded, 0, 1)

        output_padded = output_padded.reshape(N, S // 2, E * 2)

        output_padded = torch.transpose(output_padded, 0, 1)

        return output_padded, output_lengths


