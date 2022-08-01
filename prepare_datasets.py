import py_vncorenlp
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import *
from transformers import AutoTokenizer
from config import args
import numpy as np

args = args()

py_vncorenlp.download_model(save_dir=args.rdrsegmenter_path)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=args.rdrsegmenter_path)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

train_df = pd.read_csv(args.train_path)
test_df = pd.read_csv(args.test_path)

a_col = train_df.select_dtypes('int64').columns.to_list()
x_train = train_df['Review']
y_train = train_df[a_col].clip(upper=1)

x_test = test_df['Review']
y_test = test_df[a_col].clip(upper=1)

print('Cleaning:')
x_train = x_train.progress_apply(lambda x : text_cleaner(x))
x_test = x_test.progress_apply(lambda x : text_cleaner(x))

print('Word segment:')
x_train = x_train.progress_apply(lambda x: ' '.join(rdrsegmenter.word_segment(x)))
x_test = x_test.progress_apply(lambda x: ' '.join(rdrsegmenter.word_segment(x)))

print('Tokenizing:')
x_train = convert_to_feature(x_train, tokenizer, args.max_sequence_length)
x_test = convert_to_feature(x_test, tokenizer, args.max_sequence_length)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

np.save(args.data_npy, np.concatenate((x_train, x_test), axis = 0))
np.save(args.target_npy, np.concatenate((y_train, y_test), axis = 0))

print("Done")