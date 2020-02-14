from fastai.text import *
import numpy as np
import pandas as pd
import os
from os import path
import json
from pandas import json_normalize
from sklearn.utils import shuffle

training_set_name = 'ml_class.json'
DATASET_COLUMNS = ['Sentiment_Score', 'ID', 'Time', 'Query_Status', 'Account_Name', 'Tweet']
DATASET_ENCODING = "ISO-8859-1"
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
TEMP_LM_NAME = 'temp_lm'
LEARNER_NAME = 'learner'
MODEL_PATH = 'export.pkl'
dir_path = path.dirname(path.realpath(__file__))
X_COL = 'text'
Y_COL = 'labels'

'''
Training JSON data should be next format
{
  "data": [
    {
      "content": "Dynamic and filled",
      "annotation": {
        "labels": [
          "NONE"
        ],
        "note": ""
      },
      "extras": null,
      "metadata": {
        "first_done_at": 1551859592000,
        "last_updated_at": 1551859592000,
        "sec_taken": 5,
        "last_updated_by": "LK8Z68rAPZMAtNCi0g6jbLBtT5G3",
        "status": "done",
        "evaluation": "NONE"
      }
    }, 
    ....
    ....
    ]
}
'''


def prepare_vectors():
    print('Start training')
    print('TF_KERAS:', os.environ.get('TF_KERAS'))

    training_set_path = path.join(dir_path, 'data', training_set_name)
    temp_lm_path = path.join(dir_path, 'data', TEMP_LM_NAME)
    model_path = path.join(dir_path, 'models', MODEL_PATH)

    # paste dataset into data directory and name it training.csv

    with open(training_set_path) as file:
        data = json.load(file)

    norm_df = json_normalize(data['data'])
    df = norm_df[['content', 'annotation.labels']]

    # create two columns: text - with input data, and labels - labels data
    df.columns = [X_COL, Y_COL]

    # # concat labels to one string with delimiter by single char '|' to allow compatibility with fastai
    df[Y_COL] = df[Y_COL].str.join(sep='|')

    # shuffle data
    df = shuffle(df)

    # create small(micro) dataset to simulate real world situation, where no big dataset available in the company

    micro_df = df[:20]
    # split on train test datasets
    split_v = int(0.6 * len(micro_df)) + 1

    # split dataset on 80/20
    df_train, df_valid = micro_df[:split_v], micro_df[split_v:]

    print('df_train', df_train.shape)
    print('df_val', df_valid.shape)

    # Create Language Model (LM)
    # fine tune LM

    # concat all data for LM
    df_lm = pd.concat([df_train, df_valid], ignore_index=True)

    batch_size = 48

    # CASE1 - create LM with TextList

    data_lm = (TextList.from_df(df, path=dir_path, cols=X_COL)
               .split_by_rand_pct(0.1)
               .label_for_lm()
               .databunch(bs=batch_size))

    # save LM
    data_lm.save(temp_lm_path)

    learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.2)
    learn_lm.lr_find()
    learn_lm.recorder.plot(skip_end=10)


    # CASE2 - create LM with TextLMDataBunch
    # # create language model data bunch with vector representation of all unique words as tokens
    # data_lm = TextLMDataBunch.from_df(
    #     path=dir_path,
    #     train_df=df_train,
    #     valid_df=df_valid,
    #     label_cols='Sentiment_Score',
    #     text_cols='Tweet'
    # )
    #
    # data_lm.save(path.join(dir_path, 'data',  PICKLE_VECTORS))
    #
    # language_model = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    #
    # language_model.save_encoder(path.join(dir_path, 'data',  LEARNENR_PATH))
    #
    # data_class = TextClasDataBunch.from_df(
    #     path=dir_path,
    #     train_df=df_train,
    #     valid_df=df_valid,
    #     vocab=data_lm.train_ds.vocab,
    #     label_cols='Sentiment_Score',
    #     text_cols='Tweet'
    # )
    #
    # model = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.3)
    # model.load_encoder(path.join(dir_path, 'data',  LEARNENR_PATH))
    #
    # model.fit_one_cycle(1, max_lr=1e-1)
    #
    # # find good learning rate
    # model.lr_find()
    #
    # # model.recorder.plot(suggestion=True)
    #
    # model.export(path.join(dir_path, 'models', MODEL_PATH))


def predict(text):
    model = load_learner(path.join(dir_path, 'models'))
    pred = model.predict(text)

    print(pred)

    return pred


if __name__ == '__main__':
    prepare_vectors()
