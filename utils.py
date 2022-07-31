import pandas as pd
import json
import numpy as np
import pickle
import os
import torch
import re
import string
from tqdm import tqdm
tqdm.pandas()

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_to_feature(series, tokenizer, max_sequence_length):
    outputs = []
    outputs = np.zeros((len(series), max_sequence_length))
    cls_id = 0
    eos_id = 2
    pad_id = 1
    for idx, row in enumerate(series): 
        input_ids = tokenizer.encode(row)
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs
    
def text_cleaner(review):
    review = review.replace('\n', ' ')
    review = review.replace('-', ' ')
    # review = review.replace('.', '')
    review = re.sub("\s\s+" , " ", review)
    # review = re.sub(r'[^\w\s]', '', review)
    review = review.strip()
    # review = review.lower()
    return review
