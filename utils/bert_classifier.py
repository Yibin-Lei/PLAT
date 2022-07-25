import os

import torch
import torch.nn as nn


class BERTinfer(nn.Module):
    def __init__(self,
                 attack_model,
                 pretrained_dir,
                 nclasses,
                 case,
                 bert_dataset,
                 accelerator,
                 batch_size=64,
                 attack_second=False,
                 model=None):
        super(BERTinfer, self).__init__()
        # construct dataset loader
        self.case = case
        self.dataset = bert_dataset
        # construct model
        if model == None:
            if 'bert' in attack_model:
                from transformers import BertForSequenceClassification
                model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
            else:
                raise ValueError("attack_model %s is not supported." % attack_model)
        # Switch the model to eval mode.
        self.batch_size = batch_size
        self.attack_second = attack_second
        self.accelerator = accelerator
        self.model = self.accelerator.prepare(model)
        self.model.eval()

    def convert_to_cap(self, texts, marks):
        assert len(texts[0]) == len(marks)
        for i in range(len(texts)):
            for j in range(len(marks)):
                if marks[j] == 'cap':
                    texts[i][j] = texts[i][j].capitalize()
                elif marks[j] == 'upper':
                    texts[i][j] = texts[i][j].upper()
        return texts

    def get_probs_in_batch(self, texts, labels, marks=None):
        if marks is not None and self.case == 'cased':
            text = [text.split() for text in texts]
            text = [self.convert_to_cap(text, marks) for text in texts]
            text = [" ".join(text) for text in text]

        # transform text data into indices and create batches
        dataloader = self.dataset.get_batch_dataloader(texts, max_seq_length=None, labels=labels,
                                                       batch_size=self.batch_size)
        dataloader = self.accelerator.prepare(dataloader)
        all_preds = []
        with torch.no_grad():
            self.model.eval()
            for batch in dataloader:
                logits = self.model(**batch).logits.detach()
                probs = nn.functional.softmax(logits, dim=-1)
                all_preds.append(probs.cpu())

        return torch.cat(all_preds, dim=0)


    def get_prob_in_batch(self, texts, labels, marks=None):
        if marks is not None and self.case == 'cased':
            text = [text.split() for text in texts]
            text = [self.convert_to_cap(text, marks) for text in texts]
            text = [" ".join(text) for text in text]

        # transform text data into indices and create batches
        dataloader = self.dataset.get_batch_dataloader(texts, max_seq_length=None, labels=labels,
                                                       batch_size=self.batch_size)
        dataloader = self.accelerator.prepare(dataloader)
        all_preds = []
        with torch.no_grad():
            self.model.eval()
            for batch in dataloader:
                logits = self.model(**batch).logits.detach()
                probs = nn.functional.softmax(logits, dim=-1)
                selected_probs = torch.gather(probs, 1, batch["labels"].unsqueeze(1)).squeeze(1).cpu()
                all_preds.append(selected_probs)

        return torch.cat(all_preds, dim=0)