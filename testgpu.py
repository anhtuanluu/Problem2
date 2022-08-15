import torch
print(torch.cuda.is_available())

import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='C:/Users/atuan/Documents/Git/Problem2/vncorenlp')

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:/Users/atuan/Documents/Git/Problem2/vncorenlp')