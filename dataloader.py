import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours.
    '''

    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    #  Return the padded speech and text data, and the length of utterance and transcript ###
    input, target = zip(*batch_data)

    input_lens = torch.tensor([len(inp) for inp in input]).long()
    input_data = pad_sequence(input)
    target_lens = [len(tar) for tar in target]
    target_data = pad_sequence(target)

    return {
        'input_data': input_data,
        'target_lens': target_lens,
        'target_data': target_data,
        'input_lens': input_lens
    }


def collate_test(batch_data):
    #  Return padded speech and length of utterance ###
    input_lens = torch.tensor([len(inp) for inp in batch_data]).long()
    input_data = pad_sequence(batch_data)

    return {
        'input_lens': input_lens,
        'input_data': input_data
    }
