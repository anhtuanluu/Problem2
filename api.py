import json
import time
import threading
from queue import Empty, Queue
from flask import Flask, request, json
from extract_pairs import Extractor
import base64

app = Flask(__name__)
# Import model here
extractor = Extractor('label.csv') 

requestQueue = Queue()
CHECK_INTERVAL = 1
BATCH_SIZE = 10
BATCH_TIMEOUT = 2

def request_handler():
    while True:
        batch = []
        while not (
                len(batch) > BATCH_SIZE or
                (len(batch) > 0 and time.time() - batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                batch.append(requestQueue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        dateNKT = ""
        for req in batch:
            try:
                # pairs = extractor.extract_menu(req['image_name'])
                pairs = extractor.extract_menu(req['image'])
                out_status = 0
            except Exception as E:
                print(f'exception {E}')
                out_status = 1

            # out = {'pairs': pairs, 'out_status': out_status}
            out = {
                'image_name': req['image_name'],
                'predicts': [],
                'out_status': out_status
            }
            for pair in pairs:
                dct = {
                    'food_name_en': pair[2],
                    'food_name_vi': pair[0],
                    'food_price': pair[1]
                }
                out['predicts'].append(dct)
            
            req['output'] = out


threading.Thread(target=request_handler).start()

@app.route('/', methods=['GET'])
def health_check():
    return "Ok"

@app.route('/infer', methods=['POST'])
def extract_menu():
    # print(request.files)
    # data = request.get_json()
    # image_name = data['image_name']
    # encoded_img = data['image']

    image_name = request.form.get('image_name')
    encoded_img = request.form.get('image')

    # # img = request.files['image'].read()
    # encoded_img = request.files['image'].read()

    img = base64.b64decode(encoded_img)
    data = {'image': img, 'time': time.time(), 'image_name': image_name}
    requestQueue.put(data)
    response = {}
    count = 10
    while 'output' not in data and count > 0:
        time.sleep(CHECK_INTERVAL)

    if data['output']['out_status'] == 0:
        response = {
            'image_name': data['output']['image_name'],
            'infers': data['output']['predicts']
        }
        return json.dumps(response)
    else:
        response = {
            'status': 'failed'
        }
        return json.dumps(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
