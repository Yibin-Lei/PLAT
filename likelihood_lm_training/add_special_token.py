import os
from transformers import BertTokenizer, AutoConfig

''' This script will add special tokens to the tokenizer and model configuration file, which will be used in training 
the language model
'''

ORIGINAL_MODEL_PATH = 'roberta-base'
SPECIAL_TOKENS = ['<pos>', '<neg>']
OUTPUT_PATH = '../yelp_tokenizer'

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
    config = AutoConfig.from_pretrained(ORIGINAL_MODEL_PATH)

    special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    config.save_pretrained(OUTPUT_PATH)
