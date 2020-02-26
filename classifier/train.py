from fastai.text import *
import numpy as np
import pandas as pd
import os
from os import path
import json
from pandas import json_normalize
from sklearn.utils import shuffle
from fastprogress.fastprogress import IN_NOTEBOOK

IN_NOTEBOOK = True

training_set_name = 'ml_class.json'
DATASET_COLUMNS = ['Sentiment_Score', 'ID', 'Time', 'Query_Status', 'Account_Name', 'Tweet']
DATASET_ENCODING = "ISO-8859-1"
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
TEMP_LM_NAME = 'temp_lm'
LEARNER_NAME = 'learner'
LEARNER_ENCODER_NAME = 'learner'
MODEL_PATH = 'export.pkl'
dir_path = path.dirname(path.realpath(__file__))
X_COL = 'text'
Y_COL = 'labels'
SEPARATOR = '|'

# tuning params
BATCH_SIZE = 32
VAL_PERC = 0.8
MICRO_DS = 10
DO_SHUFFLE = True

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
MIN_AMOUNT = 10


# method to make balansed set
def sampling_k_elements(group, k=MIN_AMOUNT):
    if len(group) < k:
        return group
    return group.sample(k)


def get_micro_df(df):
    # delete labels which has less then 10 samples
    g = df.groupby(Y_COL)
    new_df = g.filter(lambda x: len(x) > MIN_AMOUNT)
    print(new_df[Y_COL].value_counts())

    # get balanced training micro set
    balanced = new_df.groupby(Y_COL).apply(sampling_k_elements).reset_index(drop=True)
    # return balanced
    return balanced


def prepare_vectors():
    print('Start training')
    print('TF_KERAS:', os.environ.get('TF_KERAS'))

    my_path = Path('data')
    print(my_path.ls())

    training_set_path = path.join(dir_path, 'data', training_set_name)

    with open(training_set_path) as file:
        data = json.load(file)

    norm_df = json_normalize(data['data'])
    lm_df = norm_df[['content', 'annotation.labels']]

    # create two columns: text - with input data, and labels - labels data
    lm_df.columns = [X_COL, Y_COL]

    # # concat labels to one string with delimiter by single char '|' to allow compatibility with fastai
    lm_df[Y_COL] = lm_df[Y_COL].str.join(sep=SEPARATOR)
    # change columns order
    lm_df = lm_df[[Y_COL, X_COL]]

    # show labels count to see if data balanced
    print(' *************************** See classes balance *****************************')

    print(lm_df[Y_COL].value_counts())
    print(' ****************************************************************************')

    # shuffle data
    if DO_SHUFFLE:
        lm_df = shuffle(lm_df)

    # create small(micro) dataset to simulate real world situation, where no big dataset available in the company
    print(' *************************** GENERATE MICRO dataset *****************************')

    micro_df = get_micro_df(lm_df)
    micro_df = shuffle(micro_df)

    print(micro_df)
    print(micro_df[Y_COL].value_counts())
    print(' ********************************************************************************')

    # split on train test datasets
    split_v = int(VAL_PERC * len(micro_df)) + 1

    # split dataset on 80/20
    df_train, df_valid = micro_df[:split_v], micro_df[split_v:]
    print(' *************************** INFO 1 *********************************************')

    print('micro df info', micro_df.info)
    print('micro df info', micro_df.shape)
    print('df_train', df_train.shape)
    print('df_val', df_valid.shape)
    print('lm_df shape', lm_df.shape)

    print('**********************************************************************************')

    # Create Language Model (LM)
    # fine tune LM

    # CASE1 - create LM with TextList

    print('Create Language Model from NOT LABELED dataset')

    data_lm = TextLMDataBunch.from_df(path=my_path,
                                      train_df=lm_df,
                                      valid_df=df_valid,
                                      # label_cols=Y_COL,
                                      text_cols=[X_COL]
                                      )

    print('vocab-----------------------------------------------------------')
    print(data_lm.vocab)
    print(data_lm)

    data_class = (TextList.from_df(micro_df, my_path, cols=[X_COL, Y_COL], vocab=data_lm.vocab)
                  .split_by_rand_pct(0.2)
                  .label_from_df(cols=Y_COL, label_delim='|')
                  .databunch(bs=16))

    # data_class.save('tmp_multi_label_clas.pkl')
    print(data_class)


    # *****************************************************************************************************
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

    # train LM
    learn.fit_one_cycle(1, 1e-1)
    learn.unfreeze()
    learn.fit_one_cycle(1, 1e-1)
    print('LM Predict next words:')
    print(learn.predict('very challenging in terms', n_words=10))

    # *****************************************************************************************************

    # learn CL
    learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.2, metrics=[fbeta])

    learn.fit_one_cycle(1, 1e-1, moms=(0.8, 0.7))

    # TODO: experiment with this
    # learn.freeze_to(-2)
    # learn.fit_one_cycle(1, slice(5e-3 / 2., 5e-3), moms=(0.8, 0.7))
    #
    # learn.unfreeze()
    # learn.fit_one_cycle(2, 1e-1, moms=(0.8, 0.7))
    # *****************************************************************************************************

    TEXT = 'Very challenging in terms of target and management'
    print('Test predict labels:')
    print(learn.predict(TEXT))
    print(learn.predict("it's the nature of the industry"))

    # learn.save('final_model')
    learn.export(path.join(dir_path, 'models', MODEL_PATH))

def predict(text):
    print('Load classifier')

    model = load_learner(path.join(dir_path, 'models'))
    print('Start prediction')

    pred = model.predict(text)

    print(pred)

    return pred


def text_hipo():
    path = untar_data(URLs.IMDB_SAMPLE)
    print(path)
    df = pd.read_csv(path / 'texts.csv')
    print(df.head())
    # Language model data
    data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
    # Classifier model data
    data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    learn.fit_one_cycle(1, 1e-2)
    learn.unfreeze()
    learn.fit_one_cycle(1, 1e-3)
    print('predict next words......')
    print(learn.predict("This is a review about", n_words=10))
    print(learn.predict("The management is good but", n_words=10))
    learn.save_encoder('ft_enc')
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder('ft_enc')
    # print(data_clas.show_batch())
    learn.fit_one_cycle(1, 1e-2)
    # learn.freeze_to(-2)
    # learn.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
    learn.unfreeze()
    learn.fit_one_cycle(1, slice(2e-3 / 100, 2e-3))
    print('First pred:', learn.predict("This was a great movie!"))
    print('Second pred: ', learn.predict("Well I am not sure about it...looks not great"))


if __name__ == '__main__':
    prepare_vectors()
    # text_hipo()
