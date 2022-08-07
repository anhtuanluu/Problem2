from transformers import *
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import *
from transformers import AutoTokenizer
from config import args
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(69)
args = args()

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)