import pandas as pd
import json
import numpy as np
import pickle
import os
import torch
import re
import string
from collections import OrderedDict

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def add_tail_padding(series, tokenizer, max_sequence_length):
    eos_id = 2
    pad_id = 1
    outputs = []
    outputs = np.zeros((len(series), max_sequence_length))
    for idx, row in enumerate(series): 
        input_ids = tokenizer.encode(row)
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def add_head_padding(series, tokenizer, max_sequence_length):
    eos_id = 2
    pad_id = 1
    outputs = []
    outputs = np.zeros((len(series), max_sequence_length))
    for idx, row in enumerate(series): 
        input_ids = tokenizer.encode(row)
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[len(input_ids) - (max_sequence_length - 1):]
            input_ids.append(eos_id)
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def convert_to_feature(series, tokenizer, max_sequence_length, head = False):
    if not head:
        outputs = add_tail_padding(series, tokenizer, max_sequence_length)
    else:
        outputs = add_head_padding(series, tokenizer, max_sequence_length)
    return outputs
    
def text_cleaner(review):
    review = review.replace('\n', ' ')
    review = review.replace('-', ' ')
    # review = review.replace('.', '')
    # review = re.sub(r'[^\w\s]', '', review)
    review = re.sub("\s\s+" , " ", review)
    review = review.strip()
    review = review.lower()
    return review

def load_state_dict(model, checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
