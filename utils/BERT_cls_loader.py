import os

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset

from transformers import BertTokenizer
from utils.dataloader import pad_sequence, TransformerDataset


class BERTClsDataloader(Dataset):

    def __init__(self, case):
        if case == 'cased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        elif case == 'uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            raise ValueError(case)

    def get_batch_dataloader(self, data, max_seq_length=256, labels=None, batch_size=32, shuffle=False):
        def collate_fn(data):
            data_dict = {}
            for k in list(data[0].keys()):
                if k != "labels":
                    data_dict[k] = pad_sequence([d[k] for d in data], batch_first=True,
                                                          padding_value=self.tokenizer.pad_token_id)
                else:
                    data_dict[k] = torch.tensor([d["labels"] for d in data])
            return data_dict
        tokenized_samples = self.tokenizer(
            data,
            add_special_tokens=True,
            max_length=max_seq_length,
            pad_to_max_length=False,
            return_attention_mask=True,
        )
        if labels is not None:
            tokenized_samples.data["labels"] = labels
        dataset = TransformerDataset(tokenized_samples.data)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return data_loader

    def transform_text(self, data, max_seq_length=256, labels=None,
                       batch_size=32, shuffle=False):
        # data contain list[tuple(text1, text2)]
        # transform data into seq of embeddings
        input_ids, attention_masks, token_ids = self.convert_examples_to_features(
            data, max_seq_length)

        if labels is not None:
            assert len(labels) == len(data)
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, token_ids, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks, token_ids)

        # Run prediction for full data
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        datasetloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return datasetloader

    def convert_examples_to_features(self, examples, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""
        encoded_dict = self.tokenizer.batch_encode_plus(
            examples,
            add_special_tokens=True,
            max_length=max_seq_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        return input_ids, attention_masks, token_type_ids

