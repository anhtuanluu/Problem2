import json
import numpy as np
import pickle
import os
import torch
import re
import string
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def remove_punctuation(text, PUNCT_TO_REMOVE):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text, STOPWORDS):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)
    
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

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

def add_tail_padding_text(text, tokenizer, max_sequence_length):
    eos_id = 2
    pad_id = 1
    outputs = []
    # outputs = np.zeros((len(series), max_sequence_length))
    # for idx, row in enumerate(text): 
    input_ids = tokenizer.encode(text)
    if len(input_ids) > max_sequence_length: 
        input_ids = input_ids[:max_sequence_length] 
        input_ids[-1] = eos_id
    else:
        input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
    # outputs[idx,:] = np.array(input_ids)
    return input_ids

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

def remove_long_word(text):
    return re.sub(r'([A-Z])\1+', lambda m: m.group(1).lower(), text, flags=re.IGNORECASE)

def text_cleaner(review):
    review = review.replace('\n', ' ')
    review = review.replace('.', '. ')
    review = review.replace('-', ' ')
    review = review.replace('Pig c', 'Big C')
    review = review.replace('ks', 'khách sạn')
    review = review.replace('đt', 'điện thoại')
    review = review.replace('vsinh', 'vệ sinh')
    review = review.replace('sđt', 'số điện thoại')
    review = review.replace('QN', 'Quy Nhơn')
    review = review.replace('vs', 'với')
    review = review.replace('phcu5', 'phục')
    review = review.replace('lém', 'lắm')
    # review = review.replace('fb', 'facebook')
    review = review.replace('5sao', '5 sao')
    review = review.replace('TP', 'thành phố')
    review = review.replace('tp', 'thành phố')
    review = review.replace('Tp', 'thành phố')
    review = review.replace(' 000', '000')
    review = review.replace('HN', 'Hà Nội')
    review = review.replace('Dzô', 'vô')
    review = review.replace(' ni ', ' này ')
    review = review.replace('19 k', '19k')
    review = review.replace('cf', 'coffee')
    review = review.replace(' h ', ' giờ ')
    review = review.replace('coffe', 'coffee')
    review = review.replace('tiaafn', 'tuần')
    review = review.replace('thíh', 'thích')
    review = review.replace('viawf', 'vừa')
    review = review.replace('reccomend', 'recommend')
    review = review.replace('qá', 'quá')
    review = review.replace('zễ', 'dễ')
    review = review.replace('tưoi', 'tươi')
    # review = review.replace(' k ', ' không ')
    review = review.replace('sbay', 'sân bay')
    # review = review.replace(' đ ', ' đồng ')
    review = review.replace(' a ', ' anh ')
    review = review.replace('zể', 'dễ')
    review = review.replace('<3', '')
    review = review.replace('cới', 'cái')
    review = review.replace(' r ', ' rồi ')
    review = review.replace('ncl', 'nói chung là')
    review = review.replace('lê tân', 'lễ tân')
    review = review.replace('nèeee', 'nè')
    review = review.replace('hay nan', 'hãy nán')
    review = review.replace('vnđ', 'việt nam đồng')
    review = review.replace(' oto ', 'ô tô')
    review = review.replace('trung tân', 'trung tâm')
    review = review.replace('bsang', 'buổi sáng')
    review = review.replace('dim sum', 'dimsum')
    review = review.replace('đubgs', 'đúng')
    review = review.replace('vói', 'với')
    review = review.replace('rata', 'rất')
    review = review.replace('nưa', 'nữa')
    review = review.replace('mk', 'mình')
    review = review.replace('thik', 'thích')
    review = review.replace(' ak ', ' à ')
    review = review.replace('lagu', 'lẩu')
    review = review.replace('kco', 'không có')
    review = review.replace('xug', 'xung')
    review = review.replace('gthieu', 'giới thiệu')
    review = review.replace('ík', 'ý')
    review = review.replace('ngonnn', 'ngon')
    review = review.replace('vang tau', 'Vũng Tàu')
    review = review.replace('hanghf', 'hàng')
    review = review.replace('dứoi', 'dưới')
    review = review.replace('thàn', 'thành')
    review = review.replace('dàii', 'dài')
    review = review.replace('hơnn', 'hơn')
    review = review.replace('banf', 'bàn')
    review = review.replace('tụe', 'tự')
    review = review.replace(' od ', ' order ')
    review = review.replace('rấtttt', 'rất')
    review = review.replace('nhiệtt', 'nhiệt')
    review = review.replace('thig', 'thì')
    review = review.replace('niawx', 'nữa')
    review = review.replace('thoid', 'thói')
    review = review.replace('khanggggg', 'khang')
    review = review.replace('trangggg', 'trang')
    review = review.replace('cảh', 'cả')
    review = review.replace('ltinh', 'linh tinh')
    review = review.replace('nợi', 'nơi')
    review = review.replace('nới', 'nơi')
    review = review.replace('tôt', 'tốt')
    review = review.replace('vuii', 'vui')
    review = review.replace('lạisau', 'lại sau')
    review = review.replace(' cx ', ' cũng ')
    review = review.replace(' luac ', ' lúc ')
    review = review.replace('hiẹu', 'hiệu')
    review = review.replace('ctac', 'công tác')
    review = review.replace('veiw', 'view')
    review = review.replace('trog', 'trong')
    review = review.replace(' thíc ', ' thích ')
    review = review.replace(' cacs ', ' các ')
    review = review.replace(' gogle ', ' google ')
    review = review.replace(' lăms ', ' lắm ')
    review = review.replace(' chil ', ' chill ')
    review = review.replace('thànhhh', 'thành')
    review = review.replace(' fb ', ' facebook ')
    review = review.replace(' ko ', ' không ')
    review = review.replace(' coffeeee ', ' coffee ')
    review = review.replace(' coffeee ', ' coffee ')
    review = review.replace(' coffeeeee ', ' coffee ')
    review = review.replace(' 40ng ', ' 40 người ')
    review = review.replace(' j ', ' gì ')
    review = review.replace(' tv ', ' tivi ')
    review = review.replace(' cg ', ' cũng ')
    review = review.replace(' bik ', ' biết ')
    review = review.replace(' đg ', ' đường ')
    review = remove_emoji(review)
    review = remove_emoticons(review)
    # review = remove_long_word(review) // bị sai
    # review = '<s> '+ review +' </s>'
    # review = review.lower()
    # review = re.sub(r'[^\w\s]', '', review)
    review = re.sub("\s\s+" , " ", review)
    review = re.sub("(\D) k " , "\\1 không ", review)
    review = re.sub("([0-9]) k " , "\\1 nghìn ", review)
    review = re.sub("([0-9]) đ " , "\\1 nghìn ", review)
    review = re.sub("([0-9])k " , "\\1 nghìn ", review)
    review = review.strip()
    return review

def load_state_dict(model, checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def remove_freqwords(text, FREQWORDS):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

def remove_rarewords(text, RAREWORDS):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

def remove_space_between_numbers(text):
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text

def get_new_labels(y):
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new

def r2_score_tmp(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2