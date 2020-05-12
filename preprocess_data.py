import numpy as np
import constants as con


def load_data():
    """
    Loading all the numpy files containing the utterance information and text information
    :return:
    """

    data_dict = dict()

    data_dict['speech_train'] = np.load(con.DATA_PATH + 'dev.npy', allow_pickle=True, encoding='bytes')
    data_dict['speech_valid'] = np.load(con.DATA_PATH + 'dev.npy', allow_pickle=True, encoding='bytes')
    data_dict['speech_test'] = np.load(con.DATA_PATH + 'dev.npy', allow_pickle=True, encoding='bytes')

    data_dict['transcript_train'] = np.load(con.DATA_PATH + './dev_transcripts.npy', allow_pickle=True,
                                            encoding='bytes')
    data_dict['transcript_valid'] = np.load(con.DATA_PATH + './dev_transcripts.npy', allow_pickle=True,
                                            encoding='bytes')


    return data_dict


def transform_letter_to_index(transcripts, letter_list):
    """""
    :param transcripts :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    """
    l2il = []
    for transcript in transcripts:
        transcript_list = []
        for i, word in enumerate(transcript):

            transcript_list += [letter_list.index(char.upper()) for char in word.decode('utf-8')]

            if i != len(transcript) - 1:
                transcript_list += [letter_list.index(' ')]

        l2il.append([letter_list.index('<sos>')] +
                    transcript_list + [letter_list.index('<eos>')])
    return l2il

'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()

    for i, letter in enumerate(letter_list):
        letter2index[letter] = i
        index2letter[i] = letter_list[i]
    return letter2index, index2letter


