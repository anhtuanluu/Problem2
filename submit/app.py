from flask import Flask, request, jsonify
import settings
import pickle
from utils import *

app = Flask(__name__)

loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~’‘——-'

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    review_sentence = request.args.get('review_sentence')
    review_sentence = text_cleaner(review_sentence)
    review_sentence = remove_punctuation(review_sentence, PUNCT_TO_REMOVE)
    review_sentence = review_sentence.lower()
    predict_results = loaded_model.predict([review_sentence])[0]

    output = {
        "review": review_sentence,
        "results": {}
      }
    
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = predict_results[count]

    return jsonify(output)

if __name__ == '__main__':
    app.run(host=settings.HOST, port=settings.PORT)