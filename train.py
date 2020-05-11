import torch
from dataloader import Speech2TextDataset, collate_train, collate_test
from preprocess_data import transform_letter_to_index, load_data, create_dictionaries
from torch.utils.data import DataLoader
import constants as con
from pBLSTM import pBLSTM
from Encoder import Encoder
from Seq2Seq import Seq2Seq
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import numpy as np
from Levenshtein import distance as levenshtein_distance
import csv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import time

class Train:
    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.epochs = 50
        self.optimizer = None
        self.scheduler = None

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

    def train_model_debug(self):
        for feed_dict in tqdm(self.train_loader):
            feed_dict = self.convert_to_cuda(feed_dict)
            x = feed_dict['input_data']
            x_lens = feed_dict['input_lens']
            y = feed_dict['target_data']
            y_lens = feed_dict['target_lens']

            # x.shape -> (S, N, E)
            # y.shape - > (S, N)
            # speech_input, speech_len, text_input=None, isTrain=True, epoch=None):
            predictions = self.model(speech_input=x, speech_len=x_lens, text_input=y,
                       isTrain=True, epoch=None)


    def convert_to_cuda(self, feed_dict):
        for key in feed_dict:
            if not isinstance(feed_dict[key], list):
                feed_dict[key] = feed_dict[key].to(con.DEVICE)
        return feed_dict

    def run(self):
        self.train_loader, self.val_loader, self.test_loader = self.setup()
        self.model = Seq2Seq(input_dim=40, vocab_size=len(con.LETTER_LIST), hidden_dim=512)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    patience=1,
                                                                    min_lr=1e-5,
                                                                    verbose=True)
        self.train_model_v2()

    def train_model_v2(self):
        self.model.train()
        self.model.to(DEVICE)
        start = time.time()
        for epoch in range(self.epochs):
            # 1) Iterate through your loader
            running_loss = 0
            for feed_dict in tqdm(self.train_loader):
                feed_dict = self.convert_to_cuda(feed_dict)
                x = feed_dict['input_data']
                x_lens = feed_dict['input_lens']
                y = feed_dict['target_data']
                y_lens = feed_dict['target_lens']

            # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion

            # 3) Set the inputs to the device.

            # 4) Pass your inputs, and length of speech into the model.
                predictions = self.model(speech_input=x, speech_len=x_lens, text_input=y,
                           isTrain=True, epoch=epoch)

                y = y.t()[:, 1:]
                predictions = predictions[:, :-1:, :]
                predictions = predictions.permute(0, 2, 1)

                loss = self.criterion(predictions, y)

                loss.backward()

                torch.nn.utils.clip_grad_norm(self.model.parameters(), 2)

                # 11) Take a step with your optimizer
                self.optimizer.step()

                # 12) Normalize the masked loss
                running_loss += loss

            print('Epoch: {} Loss: {} Perplexity: {}'.format(epoch + 1, running_loss / len(self.train_loader),
                                                             torch.exp(running_loss / len(self.train_loader))))
            # Saving model
            print("Saving model...")
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       'model' + str(epoch + 1) + '.pt')
            print("----------------")
            lvd = self.validation()
            self.scheduler.step(lvd)
            self.test()

    def validation(self):
        self.model.eval()
        self.model.to(DEVICE)
        start = time.time()
        letter2index, index2letter = create_dictionaries(con.LETTER_LIST)
        target = []
        preds = []
        for feed_dict in tqdm(self.val_loader):
            feed_dict = self.convert_to_cuda(feed_dict)
            x = feed_dict['input_data']
            x_lens = feed_dict['input_lens']
            y = feed_dict['target_data']
            y_lens = feed_dict['target_lens']

        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion

        # 3) Set the inputs to the device.

        # 4) Pass your inputs, and length of speech into the model.
            predictions = self.model(speech_input=x, speech_len=x_lens, text_input=y,
                       isTrain=False, epoch=None)

            y = y.t()[:, 1:]
            predictions = predictions[:, :-1:, :]

            y = y.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

            for i in range(y.shape[0]):
                res = ''
                for j in range(y.shape[1]):
                    res += index2letter[y[i][j]]
                res = res.replace('<sos>', '').replace('<pad>', '').replace('<eos>', '')
                target.append(res)


            for i in range(predictions.shape[0]):
                res = ''
                for j in range(predictions.shape[1]):
                    res += index2letter[np.argmax(predictions[i][j])]
                res = res.replace('<sos>', '').replace('<pad>', '').replace('<eos>', '')

                preds.append(res)

        lvd = 0.0
        for i in range(len(preds)):
            lvd += levenshtein_distance(target[i], preds[i])

        print('Levenshtein_distance:', lvd / len(preds))
        return lvd / len(preds)

    def test(self):
        self.model.eval()
        self.model.to(DEVICE)
        start = time.time()
        letter2index, index2letter = create_dictionaries(con.LETTER_LIST)
        preds = []
        for feed_dict in tqdm(self.test_loader):
            feed_dict = self.convert_to_cuda(feed_dict)
            x = feed_dict['input_data']
            x_lens = feed_dict['input_lens']

            predictions = self.model(speech_input=x, speech_len=x_lens, text_input=None,
                       isTrain=False, epoch=None)

            predictions = predictions[:, :-1:, :]

            predictions = predictions.detach().cpu().numpy()

            for i in range(predictions.shape[0]):
                res = ''
                for j in range(predictions.shape[1]):
                    res += index2letter[np.argmax(predictions[i][j])]
                print(res)
                res = res.replace('<sos>', '').replace('<pad>', '').replace('<eos>', '')

                preds.append(res)


        with open('submission.csv', 'w') as csvfile:
            testwriter = csv.writer(csvfile, delimiter=',')
            fieldnames = ['id', 'Predicted']
            testwriter.writerow(fieldnames)
            for i in range(len(preds)):
                testwriter.writerow([i, preds[i]])

train = Train()
train.run()

