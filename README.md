# multi-label-text-classification
Python server with Tensorflow 2.0 for multi label text classification

### NOTE!: work on docker with Fats AI env

### Start Training
#### Set bert-keras envs
``export TF_KERAS=1``

### Docker build image
``sudo docker build -t text-classify:latest .``

### Run container Detached

``sudo docker run -d -p 5000:8282 text-classify --name text-proc``


### push image
``sudo docker push serhiyskoromets/text-classify``

### floyd run training
``floyd run --data serhiyskoromets/datasets/twitter/1:classifier/data/training.csv --env pytorch-1.4 --follow "python classifier/train.py"``
