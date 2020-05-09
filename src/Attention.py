import torch
from torch import nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        '''
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Input shape of bmm:  (batch_size, max_len, hidden_size), (batch_size, hidden_size, 1)
        # Output shape of bmm: (batch_size, max_len, 1)
        attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2)

        # Create an (batch_size, max_len) boolean mask for all padding positions
        # Make use of broadcasting: (1, max_len), (batch_size, 1) -> (batch_size, max_len)
        mask = torch.arange(key.size(1)).unsqueeze(0) >= torch.tensor(lens).unsqueeze(1)
        mask = mask.to(DEVICE)

        # Set attention logits at padding positions to negative infinity.
        attention.masked_fill_(mask, -1e9)

        attention = nn.functional.softmax(attention, dim=1)

        # Compute attention-weighted sum of context vectors
        # Input shape of bmm: (batch_size, 1, max_len), (batch_size, max_len, hidden_size)
        # Output shape of bmm: (batch_size, 1, hidden_size)
        out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return out, attention