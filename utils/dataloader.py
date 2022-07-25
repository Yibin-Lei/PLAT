import os
import re
import csv
import random
from torch.utils.data import Dataset
random.seed(2020)

import numpy as np
import torch

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data[list(self.data.keys())[0]])

    def __getitem__(self, index):
        item = {}
        for k,v in self.data.items():
            item[k] = torch.tensor(v[index])
        return item

    def __len__(self):
        return self.length

def pad_sequence(sequences, batch_first=False, padding_value=0, left=False):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    if not len(sequences):
        return torch.empty(0)
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        s_slice = slice(-length, None) if left else slice(None, length)
        s_slice = (i, s_slice) if batch_first else (s_slice, i)
        out_tensor[s_slice] = tensor

    return out_tensor

def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string




def read_corpus(path, text_label_pair=False):
    with open(path, encoding='utf8') as f:
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        second_text = False if examples[1][2] == '' else True
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if not second_text:
                examples[i][2] = None
    # label, text1, text2
    if text_label_pair:
        tmp = list(zip(*examples))
        return tmp[0], list(zip(tmp[1], tmp[2]))
    else:
        return examples




