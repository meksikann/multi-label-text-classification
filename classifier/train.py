import tensorflow as tf
import os


def start_training():
    print('TF version: ', tf.__version__)
    print('Start training')
    print('TF_KERAS:', os.environ.get('TF_KERAS'))
