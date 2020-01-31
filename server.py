import tensorflow as tf
from sanic import Sanic
from sanic.response import json

print('TF version: ', tf.__version__)


app = Sanic(name='multi-label text classifier')
PORT = 8282
DEBUG = True

@app.route('/')
async def test(request):
    return json({'hello': 'world.'})


@app.route('/classify')
def classify_text(request):
    return json({'message': 'prediction'})

if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
