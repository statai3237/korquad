
# KorQuAD1.0 
This process was implemented using huggingface's transformers package.

## Dependencies
transformers == 2.11.0 \
torch >= 1.4.0
python == 3.7.0

## Usage
### simple usage 
```python
import torch 
from transformers import ElectraTokenizer
from model_qa import ElectraModel,ElectraForQuestionAnswering

config =  "monologg/koelectra-base-v3-discriminator" 
tokenizer = ElectraTokenizer.from_pretrained(config)
model = ElectraForQuestionAnswering.from_pretrained(config)
```
### Training & Evaluate
```python
$ python main.py --train_file {train_file_path} \
                 --predict_file {predict_file_path} \
                 --do_train \
                 --do_eval
```
