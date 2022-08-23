from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, r2_score
from transformers import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils import *
from config import args
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import py_vncorenlp

text = "Bánh rất nhiều tôm to, tôm giòn nằm chễm chệ trên vỏ bánh mềm thơm ngon. Món ăn thuộc loại rolling in the deep, nghĩa là cuốn với rau, dưa chuột, giá, vỏ bánh mềm. Ngoài ra, đặc biệt không thể thiếu của món ăn là nước chấm chua cay rất Bình Định, vừa miệng đến khó tả. Đặc biệt, quán có sữa ngô tuyệt đỉnh, kết hợp Combo với bánh xèo cuốn này tạo thành một cặp trời sinh. Ai không thích tôm nhảy, có thể đổi sang bò hoặc mực cũng ngon không kém."
# text = "bánh nhiều tôm to tôm giòn nằm chễm_chệ trên vỏ bánh mềm thơm ngon món ăn thuộc loại rolling in the deep nghĩa_là cuốn rau dưa_chuột giá vỏ bánh mềm ngoài_ra đặc_biệt không_thể thiếu của món ăn nước_chấm chua_cay bình_định vừa_miệng đến khó tả đặc_biệt quán sữa ngô tuyệt_đỉnh kết_hợp combo bánh_xèo cuốn này tạo thành một cặp trời sinh ai không thích tôm nhảy có_thể đổi sang bò hoặc mực ngon không kém"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seed_everything(42)
args = args()
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~’‘——“”'
FREQWORDS = set(line.strip() for line in open('freq.txt', 'r', encoding='utf-8'))
RAREWORDS = set(line.strip() for line in open('rare.txt', 'r', encoding='utf-8'))

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=args.rdrsegmenter_path, max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained("C:/Users/atuan/Documents/Git/Problem2/models")
config = AutoConfig.from_pretrained(
            "C:/Users/atuan/Documents/Git/Problem2/models",
            # output_hidden_states=True,
            output_attentions=True,
            num_labels=6
            )

mymodel = AutoModelForSequenceClassification.from_pretrained("C:/Users/atuan/Documents/Git/Problem2/models", config=config)
mymodel.to("cpu")
mymodel.eval()

data_npy = np.load(args.data_npy)
target_npy = np.load(args.target_npy)
# print(text)
text = text_cleaner(text)
text = ' '.join(rdrsegmenter.word_segment(text))
text = remove_punctuation(text, PUNCT_TO_REMOVE)
text =  text.lower()
text = text_cleaner(text)
text = re.sub("\s\s+" , " ", text).strip()
text = remove_freqwords(text, FREQWORDS)
text = remove_rarewords(text, RAREWORDS)

# print(data_npy[0].astype(int))

input_id = add_tail_padding_text(text, tokenizer, args.max_sequence_length)
input_id = torch.tensor(input_id, dtype=torch.long).unsqueeze(0)
logits = []
for i in range(5):
    checkpoint = torch.load(f"C:/Users/atuan/Documents/Git/Problem2/submit2/model_{i}.bin", map_location=torch.device('cpu'))
    load_state_dict(mymodel, checkpoint)
    logit, attentions = mymodel(input_id, attention_mask=(input_id>1), return_dict = False)
    logit = logit.view(-1).detach().cpu().numpy()
    print(logit)
    logits.append(logit)
logits = np.mean(logits,axis=0)
output = np.round(np.clip(logits, 0, 5))
print(output)