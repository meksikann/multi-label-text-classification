import tensorflow as tf

from flask import Flask

print('TF version: ', tf.__version__)

app = Flask(__name__)

PORT = 8282

@app.route('/classify')
def classify_text():
    return 'Hey, we have Flask in a Docker container!'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)

