import py_vncorenlp
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='C:/Users/atuan/Documents/Git/Problem2/vncorenlp')

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:/Users/atuan/Documents/Git/Problem2/vncorenlp')

text = "Thực phẩm tốt nhưng đắt tiền cho những gì bạn nhận được"

output = tokenizer.encode_plus(' '.join(rdrsegmenter.word_segment(text)), padding='max_length', truncation=True, max_length=256, return_attention_mask=True)

print(output)
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']