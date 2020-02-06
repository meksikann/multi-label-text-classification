from sanic import Sanic
from sanic.response import json, text

from classifier.train import predict

app = Sanic(name='multi-label text classifier')

predictor_settings = {
    'DATASET_NAME': 'data.csv',
}
app.config.update(predictor_settings)

PORT = 8282
DEBUG = True

@app.route('/')
async def test(request):
    return json({'hello': 'world.'})


@app.route('/train-predictor', methods=['POST'])
def train_predictor(request):
    start_training()

    print('send response')
    return json({'message': 'Training started ....'})


@app.route('/classify', methods=['POST'])
def classify_text(request):
    body = request.json
    print(body)

    res = predict(body['text'])

    return text(res)

if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
