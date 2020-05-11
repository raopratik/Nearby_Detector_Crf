import torch
from torch import nn
from Attention import Attention
from constants import DEVICE
import random

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step,
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''

    def __init__(self, vocab_size, hidden_dim=512, value_size=128, key_size=128, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
        self.teacher_prob = 0.0
        self.curr_epoch = 0
        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, isTrain=True, epoch=None):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[1]

        if isTrain and self.teacher_prob < 0.5 and self.curr_epoch != epoch:
            self.teacher_prob += 0.025
            self.curr_epoch = epoch

        if (isTrain == True):

            max_len = text.shape[0]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, 35).to(DEVICE)

        context_vector = values[0, :, :]


        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break
            #   out of the loop so you do you do not get index out of range errors.
            curr_prob = random.uniform(0, 1)
            if i == 0:
                prediction[:, 33] = 1

            if (isTrain) and curr_prob >= self.teacher_prob:
                char_embed = embeddings[i, :, :]
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            # print("char_emb", char_embed.shape, "context", context_vector.shape)
            inp = torch.cat([char_embed, context_vector], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]

            context_vector, attention = self.attention(query=output, key=key,
                                                       value=values, lens=lens)

            prediction = self.character_prob(torch.cat([output, context_vector], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)
