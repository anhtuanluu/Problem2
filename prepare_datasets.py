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
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=args.rdrsegmenter_path, max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
STOP_WORDS = set()
with open("E:\Git\Problem2/data/stopword_dash.txt", "r", encoding="utf8") as f:
    for i in f.readlines():
        STOP_WORDS.add(i.strip())

train_df = pd.read_csv(args.train_path)

a_col = train_df.select_dtypes('int64').columns.to_list()
x_train = train_df['Review']
# y_train = train_df[a_col].clip(upper=1)
y_train = train_df[a_col]
x_train = x_train.progress_apply(lambda x: ' '.join(rdrsegmenter.word_segment(x)))
x_train = x_train.apply(lambda x: remove_punctuation(x, PUNCT_TO_REMOVE))
x_train = x_train.progress_apply(lambda x : text_cleaner(x))
x_train = x_train.apply(lambda x: '<s> '+ x +' </s>')
print(x_train[1])
print('Tokenizing:')
x_train = convert_to_feature(x_train, tokenizer, args.max_sequence_length, args.head)
for i in range(5):
    print(tokenizer.decode(x_train[i]))
y_train = y_train.to_numpy()

np.save(args.data_npy, x_train)
np.save(args.target_npy, y_train)

print("Done")