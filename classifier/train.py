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
LEARNER_ENCODER_NAME = 'learner'
MODEL_PATH = 'export.pkl'
dir_path = path.dirname(path.realpath(__file__))
X_COL = 'text'
Y_COL = 'labels'

# tuning params
BATCH_SIZE = 32
VAL_PERC = 0.8
MICRO_DS = 200

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

    my_path = Path('data')

    training_set_path = path.join(dir_path, 'data', training_set_name)

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
    print('Make micro dataset')
    micro_df = df[:MICRO_DS]
    # split on train test datasets
    split_v = int(VAL_PERC * len(micro_df)) + 1

    # split dataset on 80/20
    df_train, df_valid = micro_df[:split_v], micro_df[split_v:]

    print('df_train', df_train.shape)
    print('df_val', df_valid.shape)

    # Create Language Model (LM)
    # fine tune LM

    # concat all data for LM
    df_lm = pd.concat([df_train, df_valid], ignore_index=True)

    # CASE1 - create LM with TextList

    print('Create Language Model from dataset')
    data_lm = (TextList.from_df(df_lm, path=my_path, cols=X_COL)
               .split_by_rand_pct(0.1)
               .label_for_lm()
               .databunch(bs=BATCH_SIZE))

    # save LM
    data_lm.save('tmp_lm')

    print('Learn Language Model')

    learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.2)
    learn_lm.lr_find()
    # learn_lm.recorder.plot()

    learn_lm.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

    # save trained learner
    learn_lm.save('fit_head')
    learn_lm.load('fit_head')

    # learn other layers
    learn_lm.unfreeze()
    learn_lm.fit_one_cycle(20, 2e-5, moms=(0.8, 0.7))

    # plot learning results
    # learn_lm.recorder.plot_losses()
    print('Save Language Model')

    # save trained learner and encoder
    learn_lm.save('fine_tuned')
    learn_lm.save_encoder('fine_tuned_enc')

    '''
    ********************  Classifier ***********************************
    '''
    print('Build classifier')
    print('Load data')

    # class_data_lm = TextLMDataBunch.load(my_path, 'tmp_lm', bs=BATCH_SIZE)
    class_data_lm = load_data(my_path, 'tmp_lm', bs=BATCH_SIZE)

    # functions for metricts
    def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
        pred_pos = (log_preds > thresh).float()
        tpos = torch.mul((targs == pred_pos).float(), targs.float())
        return (tpos.sum() / (pred_pos.sum() + epsilon))  # .item()

    def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
        pred_pos = (log_preds > thresh).float()
        tpos = torch.mul((targs == pred_pos).float(), targs.float())
        return (tpos.sum() / (targs.sum() + epsilon))

    data_class = TextClasDataBunch.from_df(
        my_path,
        train_df=df_train,
        valid_df=df_valid,
        vocab=class_data_lm.vocab,
        text_cols='text',
        label_cols='labels',
        label_delim='|',
        bs=BATCH_SIZE
    )

    data_class.save('tmp_class')
    print('Init classifier learner')

    learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.3)

    # define metrics
    # learn.metrics = [accuracy_thresh, precision, recall]

    learn.load_encoder('fine_tuned_enc')
    print('Train classifier learner')

    learn.freeze()
    learn.fit_one_cycle(1, 2e-5, moms=(0.8, 0.7))

    # save first part trained learner
    learn.save('first_factors')
    learn.load('first_factors')

    learn.freeze_to(-2)
    learn.fit_one_cycle(2, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7), wd=0.1)

    learn.save('second_factors')
    learn.load('second_factors')

    learn.unfreeze()
    learn.fit_one_cycle(2, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7), wd=0.1)
    print('Classifier done. Export final model.')

    learn.export(path.join(dir_path, 'models', MODEL_PATH))

    # Evaluate results
    print('Build training metrics')

    y_pred, y_true = learn.get_preds()
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    from sklearn.metrics import classification_report
    f1_score(y_true, y_pred > 0.35, average='micro')
    y_true = y_true.numpy()
    scores = y_pred.numpy()

    print(scores.shape, y_true.shape)
    metrics = classification_report(y_true, scores > 0.35, target_names=data_class.valid_ds.classes)
    print(metrics)


def predict(text):
    print('Load classifier')

    model = load_learner(path.join(dir_path, 'models'))
    print('Start prediction')

    pred = model.predict(text)

    print(pred)

    return pred


if __name__ == '__main__':
    prepare_vectors()
