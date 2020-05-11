import torch
CUDA = True
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+',
               ' ','<sos>','<eos>']
DATA_PATH = 'src/data/'
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'