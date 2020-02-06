from fastai.text import *
import numpy as np
import pandas as pd
import os
from os import path

training_set_name = 'training.csv'
DATASET_COLUMNS = ['Sentiment_Score', 'ID', 'Time', 'Query_Status', 'Account_Name', 'Tweet']
DATASET_ENCODING = "ISO-8859-1"
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
PICKLE_VECTORS = 'vectors.pkl'
LEARNENR_PATH = 'learner'
MODEL_PATH = 'export.pkl'
dir_path = path.dirname(path.realpath(__file__))


def prepare_vectors():
    print('Start training')
    print('TF_KERAS:', os.environ.get('TF_KERAS'))

    training_set_path = path.join(dir_path, 'data', training_set_name)

    # paste dataset into data directory and name it training.csv

    df = pd.read_csv(training_set_path, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

    df = df.iloc[np.random.permutation(len(df))]

    df = df[:500]
    print(df.info)
    cut1 = int(0.8 * len(df)) + 1

    # split dataset on 80/20
    df_train, df_valid = df[:cut1], df[cut1:]

    # create language model data bunch with vector representation of all unique words as tokens
    data_lm = TextLMDataBunch.from_df(
        path=dir_path,
        train_df=df_train,
        valid_df=df_valid,
        label_cols='Sentiment_Score',
        text_cols='Tweet'
    )

    data_lm.save(path.join(dir_path, 'data',  PICKLE_VECTORS))

    language_model = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

    language_model.save_encoder(path.join(dir_path, 'data',  LEARNENR_PATH))

    data_class = TextClasDataBunch.from_df(
        path=dir_path,
        train_df=df_train,
        valid_df=df_valid,
        vocab=data_lm.train_ds.vocab,
        label_cols='Sentiment_Score',
        text_cols='Tweet'
    )

    model = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.3)
    model.load_encoder(path.join(dir_path, 'data',  LEARNENR_PATH))

    model.fit_one_cycle(1, max_lr=1e-1)

    # find good learning rate
    model.lr_find()

    # model.recorder.plot(suggestion=True)

    model.export(path.join(dir_path, 'models', MODEL_PATH))


def predict(text):
    model = load_learner(path.join(dir_path, 'models'))
    pred = model.predict(text)

    print(pred)

    return pred


if __name__ == '__main__':
    prepare_vectors()
