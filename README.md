# Phrase-level Textual Adversarial Attack with Label Preservation
### Introduction
This is the official code for NAACL-HLT 2022 Findings paper "Phrase-level Textual Adversarial Attack with Label Preservation".

### Preparation

1. Python dependency: The project is based on Python 3.7 and PyTorch 1.8.1. Please check `requirements.txt` to install the 
libraries needed.
2. Datasets: Please check the directory `./data/`, there are 1,000 samples for each dataset. You can use other datasets but
please ensure that you have converted them into the proper format.
3. StanfordNLP: Download [StanfordNLP](https://stanfordnlp.github.io/CoreNLP/) and unzip it under the root directory (We
use the version 4.2.2).
4. Classifier: please prepare and put your classifier under the directory `./classifier/${task}/`, where `${task}` denotes one of the four tasks
   (`yelp`, `ag`, `mnli` and `qnli`).
5. Language model for likelihood estimation. Please prepare and put your language model (e.g., Roberta) for likelihood 
estimation under the directory `./likelihood_model/${task}/`.    
We provide some training example scripts under `./likelihood_lm_training`:  
(1) For document-level examples (like `yelp` and `ag` tasks, please first break them into single sentences. For 
sentence-level tasks, we keep the whole example and skip step (2).  
(2) Given the single sentences, we use a finetuned classifier to predict the labels of them, only sentence with high 
confidence score can be kept for further training usage.  
(3) Moreover, a special token, e.g. <pos> for positive, <neg> for negative will be prepended to the beginning of each 
sentence. For sentence pair tasks like MNLI, we form the data like `<con>premesis</s></s>hypothesis`, where `<con>`
is a special token representing the label and `</s></s>` is a separation symbol in RoBERTa.  
`Steps (2) and (3) are done in "prepare_data.py.`  
(4) Then we add special tokens <pos> and <neg> to the tokenizer vocabulary, which is done by `add_special_token.py`  
(5) Finally, given the prepared training data and tokenizer, we masked languge modeling finetune them using `run_mlm.py` 
[script provided by 
Huggingface](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling). 
An example is given in `training_script.sh`


### Attack

Run:
```
python bert_attack_classification.py --dataset ${dataset} 
```
and it will save the results under the root directory. 

You can also modify the attacking hyper-parameters in `./utils/config.py` to adjust the trade-off between different aspects.


### Evaluation
We provide evaluation script for evaluating generated example in `./evaluation.py`

### Acknowledge
Some of our code are built upon [CLARE](https://github.com/cookielee77/CLARE).

### Citing

If you find our work is useful in your research, please consider citing:
```
@InProceedings{lei2022plat,
  title={Phrase-level Textual Adversarial Attack with Label Preservation},
  author={Lei, Yibin and Cao, Yu and Li, Dianqi and Zhou, Tianyi and Fang, Meng and Pechenizkiy, Mykola},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  year={2022}
}
```
