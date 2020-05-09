import torch
from dataloader import Speech2TextDataset, collate_train, collate_test
from preprocess_data import transform_letter_to_index, load_data
from torch.utils.data import DataLoader
import constans as con

class Train:
    def setup(self):
        speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
        character_text_train = transform_letter_to_index(transcript_train, con.LETTER_LIST)
        character_text_valid = transform_letter_to_index(transcript_valid, con.LETTER_LIST)

        train_dataset = Speech2TextDataset(speech_train, character_text_train)
        # val_dataset =
        test_dataset = Speech2TextDataset(speech_test, None, False)

        train_loader = DataLoader(train_dataset, batch_size=con.BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_train)
        # val_loader =
        test_loader = DataLoader(test_dataset, batch_size=con.BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_test)

        return train_loader, test_loader

    def run(self):
        train_loader, test_loader = self.setup()


