from flask import Flask, request, jsonify
import settings
import pickle
from utils import *
import py_vncorenlp

app = Flask(__name__)

loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~’‘——“”'
FREQWORDS = set(line.strip() for line in open('freq.txt', 'r', encoding='utf-8'))
RAREWORDS = set(line.strip() for line in open('rare.txt', 'r', encoding='utf-8'))
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="/bai2/vncorenlp", max_heap_size='-Xmx500m')

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    review_sentence = request.args.get('review_sentence')
    text = text_cleaner(review_sentence)
    text = ' '.join(rdrsegmenter.word_segment(text))
    text = remove_punctuation(text, PUNCT_TO_REMOVE)
    text =  text.lower()
    text = text_cleaner(text)
    text = re.sub("\s\s+" , " ", text).strip()
    text = remove_freqwords(text, FREQWORDS)
    text = remove_rarewords(text, RAREWORDS)
    
    predict_results = loaded_model.predict([text])[0]

    output = {
        "review": review_sentence,
        "results": {}
      }
    
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = predict_results[count]

    return jsonify(output)

if __name__ == '__main__':
    app.run(host=settings.HOST, port=settings.PORT)