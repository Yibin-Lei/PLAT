# Phrase-level Textual Adversarial Attack with Label Preservation
### Introduction
This is the official code for NAACL 2022 Findings paper "Phrase-level Textual Adversarial Attack with Label Preservation". 
Code is still in process.

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
estimation under the directory `./likelihood_model/${task}/`. (We are still working on this part) 

### Attack

Run:
```
python bert_attack_classification.py --dataset ${dataset} 
```
and it will save the results under the root directory. 

You can modify the attacking hyper-parameters in the command according to definitions in
 `./utils/config.py` to adjust the trade-off between different aspects.


### Evaluation


### Acknowledge
Some of our code are built upon [CLARE](https://github.com/cookielee77/CLARE).

### Citing

If you find our work is useful in your research, please consider citing:
```
@InProceedings{li2021contextualized,
  title={Phrase-level Textual Adversarial Attack with Label Preservation},
  author={Lei, Yibin and Cao, Yu and Li, Dianqi and Zhou, Tianyi and Fang, Meng and Pechenizkiy, Mykola},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  year={2022}
}
```