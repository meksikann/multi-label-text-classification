from fastai.text import *
import numpy as np
import pandas as pd
import os
from os import path
import json
from pandas import json_normalize
from sklearn.utils import shuffle
from fastprogress.fastprogress import IN_NOTEBOOK
IN_NOTEBOOK=True

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
MIN_AMOUNT = 9

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

    training_set_path = path.join(dir_path, 'data', training_set_name)

    with open(training_set_path) as file:
        data = json.load(file)

    norm_df = json_normalize(data['data'])
    lm_df = norm_df[['content', 'annotation.labels']]

    # create two columns: text - with input data, and labels - labels data
    lm_df.columns = [X_COL, Y_COL]

    # # concat labels to one string with delimiter by single char '|' to allow compatibility with fastai
    lm_df[Y_COL] = lm_df[Y_COL].str.join(sep=SEPARATOR)

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
    # data_lm = (TextList.from_df(lm_df, path=my_path, cols=X_COL)
    #            .split_by_rand_pct(0.1)
    #            .label_for_lm()
    #            .databunch(bs=BATCH_SIZE))

    # # csv_path = untar_data(URLs.IMDB_SAMPLE)
    #
    # data_lm = TextLMDataBunch.from_csv(URLs.IMDB_SAMPLE, 'texts.csv')
    data_lm = TextLMDataBunch.from_df(path=my_path,
                                      train_df=lm_df,
                                      valid_df=df_valid,
                                      label_cols=Y_COL,
                                      text_cols=X_COL
                                      )

    data_class = TextClasDataBunch.from_df(
        path=my_path,
        train_df=df_train,
        valid_df=df_valid,
        vocab=data_lm.train_ds.vocab,
        text_cols=X_COL,
        label_cols=Y_COL,
        label_delim=SEPARATOR,
        bs=32
    )

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    learn.fit_one_cycle(1, 1e-2)
    learn.unfreeze()
    learn.fit_one_cycle(1, 1e-3)
    learn.save_encoder('ft_enc')

    learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder('ft_enc')
    learn.fit_one_cycle(1, 1e-2)
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

    TEXT = 'Very challenging in terms of target and management'
    print(learn.predict(TEXT))

    learn.save('final_model')
    learn.export(path.join(dir_path, 'models', MODEL_PATH))



    # print('SHow batch of LM')
    # print(data_lm)
    #
    # # save LM
    # data_lm.save('tmp_lm')
    #
    # print('Learn Language Model')
    #
    # learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.2)
    #
    # learn_lm.lr_find()
    # # learn_lm.recorder.plot(suggestion=True)
    #
    # learn_lm.fit_one_cycle(1, 1e-1, moms=(0.8, 0.7))
    #
    # # save trained learner
    # learn_lm.save('fit_head')
    # learn_lm.load('fit_head')
    #
    # # learn other layers
    # learn_lm.unfreeze()
    # learn_lm.fit_one_cycle(10, 1e-1, moms=(0.8, 0.7))
    #
    # # plot learning results
    # # learn_lm.recorder.plot_losses()
    # print('Save Language Model')
    #
    # # save trained learner and encoder
    # learn_lm.save('fine_tuned')
    # learn_lm.save_encoder('fine_tuned_enc')
    #
    # '''
    # ********************  Classifier ***********************************
    # '''
    # print('Build classifier')
    # print('Load data')
    #
    # # class_data_lm = TextLMDataBunch.load(my_path, 'tmp_lm', bs=BATCH_SIZE)
    # class_data_lm = load_data(my_path, 'tmp_lm', bs=BATCH_SIZE)
    #
    #
    #
    # data_class.save('tmp_class')
    # print('Init classifier learner')
    #
    # learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.3)
    #
    # # define metrics
    # # learn.metrics = [accuracy_thresh, precision, recall]
    #
    # learn.load_encoder('fine_tuned_enc')
    # print('Train classifier learner')
    #
    # learn.lr_find()
    # # learn.recorder.plot(suggestion=True)
    #
    # # learn.freeze()
    # learn.fit_one_cycle(2, 1e-1, moms=(0.8, 0.7))
    #
    # # save first part trained learner
    # learn.save('first_factors')
    # learn.load('first_factors')
    #
    # learn.freeze_to(-2)
    # learn.fit_one_cycle(2, 1e-1, moms=(0.8, 0.7), wd=0.1)
    #
    # learn.save('second_factors')
    # learn.load('second_factors')
    #
    # # learn.unfreeze()
    # # learn.fit_one_cycle(2, 1e-1, moms=(0.8, 0.7), wd=0.1)
    # # print('Classifier done. Export final model.')
    #
    # TEXT = 'Very challenging in terms of target and management'
    # print(learn.predict(TEXT))
    #
    # learn.export(path.join(dir_path, 'models', MODEL_PATH))

    # Evaluate results
    # print('Build training metrics')
    #
    # y_pred, y_true = learn.get_preds()
    # from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    # from sklearn.metrics import classification_report
    # f1_score(y_true, y_pred > 0.35, average='micro')
    # y_true = y_true.numpy()
    # scores = y_pred.numpy()
    #
    # print(scores.shape, y_true.shape)
    # metrics = classification_report(y_true, scores > 0.35, target_names=data_class.valid_ds.classes)
    # print(metrics)


def predict(text):
    print('Load classifier')

    model = load_learner(path.join(dir_path, 'models'))
    print('Start prediction')

    pred = model.predict(text)

    print(pred)

    return pred


if __name__ == '__main__':
    prepare_vectors()

# TODO: hello world example
# path = untar_data(URLs.IMDB_SAMPLE)
#
# data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
# data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
#
# learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
# learn.fit_one_cycle(1, 1e-2)
# learn.unfreeze()
# learn.fit_one_cycle(1, 1e-3)
# learn.save_encoder('ft_enc')
#
# learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
# learn.load_encoder('ft_enc')
# learn.fit_one_cycle(1, 1e-2)
# learn.freeze_to(-2)
# learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
# learn.save('model')
#
# learn.predict("This was a great movie!")
