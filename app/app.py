from flask import Flask , render_template, request

import os
import base64
from io import BytesIO

from PIL import Image
import pandas as pd
from petastorm import make_reader


# APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
# FOLDER_PATH = os.path.join(APP_ROOT, 'tmp')
DEFAULT_MNIST_DATA_PATH = '/Users/PC-1/Downloads/AI-Immune-System/tmp'

app = Flask(__name__)

@app.route('/api/v1/datasets')
def list():
    files = os.listdir(DEFAULT_MNIST_DATA_PATH)
    return render_template('index.html', files=files, fileName='')

@app.route('/api/v1/dataset/<string:name>')
def show(name):
    # path = os.path.join(DEFAULT_MNIST_DATA_PATH)
    files = os.listdir(DEFAULT_MNIST_DATA_PATH)

    with make_reader(f'file://{DEFAULT_MNIST_DATA_PATH}/{name}') as reader:
        data = []
        for row in reader:
            image = Image.fromarray(row.image.astype("uint8"))
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
            img_str = '<img src="data:image/png;base64,{0}">'.format(img_str)
            data.append({ 'digit': row.digit, 'image': img_str })
        return render_template('index.html', files=files, fileName=name, data=pd.DataFrame(data).to_html(escape=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)