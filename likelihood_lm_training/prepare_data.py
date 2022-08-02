from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from nltk import sent_tokenize
import torch

''' This script aims to generate the training samples for the following likelihood language model, where only samples
with high confidence will be remained
'''
# This example model can be downloaded from https://huggingface.co/VictorSanh/roberta-base-finetuned-yelp-polarity
INPUT_FILE = '../data/yelp.tsv'
PRETRAINED_MODEL_PATH = '../VictorSanh/roberta-base-finetuned-yelp-polarity'
OUTPUT_FILE = '/yelp_LM.txt'
SPLIT_SENTENCE = True
FILTER_TH = 0.99

if __name__ == "__main__":

    f_write = open(OUTPUT_FILE, 'w')

    # pipeline for sentiment analysis

    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

    # selection for yelp
    n = 0
    with torch.no_grad():
        # read each line and make predictions
        if SPLIT_SENTENCE:
            texts = []
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
            for line in lines:
                texts.extend(sent_tokenize(line.strip().split('\t')[1]))
        else:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
            texts = [line.strip().split('\t')[1] for line in lines]
        with nlp.device_placement():
            sampler = SequentialSampler(texts)
            dataloader = DataLoader(texts, sampler=sampler, batch_size=256)
            for text in tqdm(dataloader):
                class_result = nlp(text)
                for i, line in enumerate(text):
                    score = class_result[i]["score"]
                    label = class_result[i]["label"]
                    if label == "LABEL_0" and score > FILTER_TH:
                        f_write.write("<neg> " + line)
                    if label =="LABEL_1" and score > FILTER_TH:
                        f_write.write("<pos> " + line)
