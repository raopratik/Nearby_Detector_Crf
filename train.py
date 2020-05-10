import torch
from dataloader import Speech2TextDataset, collate_train, collate_test
from preprocess_data import transform_letter_to_index, load_data
from torch.utils.data import DataLoader
import constants as con
from pBLSTM import pBLSTM
from Encoder import Encoder
from Seq2Seq import Seq2Seq
from tqdm import tqdm

class Train:
    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None

    def setup(self):
        data_dict = load_data()
        character_text_train = transform_letter_to_index(data_dict['transcript_train'], con.LETTER_LIST)
        character_text_valid = transform_letter_to_index(data_dict['transcript_valid'], con.LETTER_LIST)

        train_dataset = Speech2TextDataset(data_dict['speech_train'], character_text_train)
        val_dataset = Speech2TextDataset(data_dict['speech_valid'], character_text_valid)
        test_dataset = Speech2TextDataset(data_dict['speech_test'], None, False)

        train_loader = DataLoader(train_dataset, batch_size=con.BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_train)
        val_loader = DataLoader(val_dataset, batch_size=con.BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_train)
        test_loader = DataLoader(test_dataset, batch_size=con.BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_test)

        return train_loader, val_loader, test_loader

    def train_model(self):
        for feed_dict in tqdm(self.train_loader):
            feed_dict = self.convert_to_cuda(feed_dict)
            x = feed_dict['input_data']
            x_lens = feed_dict['input_lens']
            y = feed_dict['target_data']
            y_lens = feed_dict['target_lens']

            # x.shape -> (S, N, E)
            # y.shape - > (S, N)
            # speech_input, speech_len, text_input=None, isTrain=True, epoch=None):
            self.model(speech_input=x, speech_len=x_lens, text_input=y,
                       isTrain=True, epoch=None)


    def convert_to_cuda(self, feed_dict):
        for key in feed_dict:
            if not isinstance(feed_dict[key], list):
                feed_dict[key] = feed_dict[key].to(con.DEVICE)
        return feed_dict

    def run(self):
        self.train_loader, self.val_loader, self.test_loader = self.setup()
        self.model = Seq2Seq(input_dim=40, vocab_size=len(con.LETTER_LIST), hidden_dim=10)
        self.train_model()

train = Train()
train.run()

