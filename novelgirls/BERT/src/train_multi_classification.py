import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

import bert
from bert import run_classifier
from novelgirls.BERT.src.model import model_fn_builder
from novelgirls.BERT.src.create_tokenizer import create_tokenizer_from_hub_module as create_tokenizer

''' 모델 저장할 경로 입력'''
OUTPUT_DIR = ''

DO_DELETE = False

if DO_DELETE:
    try:
        tf.gfile.DeleteRecursively(OUTPUT_DIR)
    except:
        pass

tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

train = pd.read_excel("./BERT/Datasets/Data_Train.xlsx").loc[:, ['description', 'eye_color']]
train['eye_color'] = train['eye_color'].replace(['blue', 'brown', 'gray', 'black', 'green', 'hazel', 'yellow', 'idk'],
                                                [0, 1, 2, 3, 4, 5, 6, 7])

test = pd.read_excel("./BERT/Datasets/Data_Test.xlsx").loc[:, ['description', 'eye_color']]
test['eye_color'] = test['eye_color'].replace(['blue', 'brown', 'gray', 'black', 'green', 'hazel', 'yellow', 'idk'],
                                              [0, 1, 2, 3, 4, 5, 6, 7])

# train = pd.read_excel("./BERT/Datasets/Data_Train.xlsx").loc[:,['description', 'hair_color']]
# train['hair_color']=train['hair_color'].replace(['brown', 'red', 'gray', 'black', 'blonde', 'white', 'orange', 'idk'],
#                                                   [0, 1, 2, 3, 4, 5, 6, 7])
#
# test = pd.read_excel("./BERT/Datasets/Data_Test.xlsx").loc[:,['description', 'hair_color']]
# test['hair_color'] = test['hair_color'].replace(['brown', 'red', 'gray', 'black', 'blonde', 'white', 'orange', 'idk'],
#                                                 [0, 1, 2, 3, 4, 5, 6, 7])


train, val = train_test_split(train, test_size=0.1, random_state=100)

print("Training Set Shape :", train.shape)
print("Validation Set Shape :", val.shape)
print("Test Set Shape :", test.shape)

DATA_COLUMN = 'description'
LABEL_COLUMN = 'eye_color'
# LABEL_COLUMN = 'hair_color'
label_list = [0, 1, 2, 3, 4, 5, 6, 7]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                             text_a=x[DATA_COLUMN],
                                                                             text_b=None,
                                                                             label=x[LABEL_COLUMN]), axis=1)

val_InputExamples = val.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                         text_a=x[DATA_COLUMN],
                                                                         text_b=None,
                                                                         label=x[LABEL_COLUMN]), axis=1)

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

tokenizer = create_tokenizer(BERT_MODEL_HUB)


MAX_SEQ_LENGTH = 200
BATCH_SIZE = 16
LEARNING_RATE = 3e-5    # eye
# LEARNING_RATE = 2e-5    # hair
NUM_TRAIN_EPOCHS = 15.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 300
SAVE_SUMMARY_STEPS = 100

train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features = bert.run_classifier.convert_examples_to_features(val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR,
                                    save_summary_steps=SAVE_SUMMARY_STEPS,
                                    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(num_labels=len(label_list),
                            learning_rate=LEARNING_RATE,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps,
                            bert_model_hub=BERT_MODEL_HUB)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE})

train_input_fn = bert.run_classifier.input_fn_builder(features=train_features, seq_length=MAX_SEQ_LENGTH,
                                                      is_training=True, drop_remainder=False)
val_input_fn = run_classifier.input_fn_builder(features=val_features, seq_length=MAX_SEQ_LENGTH,
                                               is_training=False, drop_remainder=False)

print(f'***** Beginning Training! *****')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("***** Training took time ", datetime.now() - current_time, "*****")

estimator.evaluate(input_fn=val_input_fn, steps=None)


def get_prediction(in_sentences):
    labels = ['blue', 'brown', 'gray', 'black', 'green', 'hazel', 'yellow', 'idk']  # eye
    # labels = ['brown', 'red', 'gray', 'black', 'blonde', 'white', 'orange', 'idk']    # hair
    labels_list = [0, 1, 2, 3, 4, 5, 6, 7]
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
    input_features = run_classifier.convert_examples_to_features(input_examples, labels_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)

    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


pred_sentences = list(test['description'])
prediction = get_prediction(pred_sentences)
print("ex:", prediction[3])
