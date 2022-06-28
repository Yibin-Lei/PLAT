# Phrase-level Textual Adversarial Attack with Label Preservation
### Introduction
This is the official code for NAACL 2022 Findings paper "Phrase-level Textual Adversarial Attack with Label Preservation". 
Code is still in process.

### Preparation

1. Python dependency: The project is based on Python 3.8 and PyTorch. Please check `requirements.txt` and install the 
library needed.
2. Datasets: Please check the directory `data`, there are 1,000 samples for each dataset. You can use other datasets but
please ensure that you have converted them into the proper format.
3. StanfordNLP: Download [StanfordNLP](https://stanfordnlp.github.io/CoreNLP/) and unzip it under the root directory (We
use the version 4.2.2).

### Obtain the parsed phrases for attacking

Execute `extract_phrases.py` under the root directory, which will generated the parsed phrases of a dataset under 
`parsed_data`. An example is like

`python extract_phrase.py --input_file data/ag.tsv --output_prefix ag`