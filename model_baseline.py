# This file implements a baseline model for cassava leaf disease classification

import time
import pickle
import numpy as np
import pandas as pd
import shutil, os
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, Input
import json
import tensorflow as tf
from scipy import stats
from tensorflow.data import TFRecordDataset
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import VGG16
from matplotlib import pyplot as plt
# from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image, ImageStat
from skimage import io, color

# Loading the training data
train_raw = pd.read_csv('train.csv', encoding='utf_8_sig', engine='python')
print(train_raw.head())

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
TRAIN_PATH = 'train_tfrecords/'
IMAGE_SIZE = [512, 512]
###


def _parse_function(proto):
    # feature_description needs to be defined since datasets use graph-execution
    # - its used to build their shape and type signature
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.cast(image, tf.float32) # :: [0.0, 255.0]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    target = tf.one_hot(parsed_features['target'], depth=5)
    image_id = parsed_features['image_name']
    return image, target, image_id


def _preprocess_fn(image, label, image_id):
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    label = tf.concat([label, [0]], axis=0)
    return image, label, image_id


def load_dataset(file_names):
    raw_ds = tf.data.TFRecordDataset(file_names)
    parsed_ds = raw_ds.map(_parse_function)
    parsed_ds = parsed_ds.map(_preprocess_fn)
    return parsed_ds


def build_valid_ds(valid_f):
    ds = load_dataset(valid_f)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds


train_files = [TRAIN_PATH + f for f in os.listdir(TRAIN_PATH)]


build_valid_ds(train_files)
###

def save_obj(obj, name):
    """
    Save an object to local, using pickle
    :param obj: ...
    :param name: ...
    :return: None
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loading an local object using pickle
    :param name: ...
    :return: Loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


_batch_size = 32
_image_width = 160
_image_height = 120


def file_move(file_names, labels, train=True, pic_file='D:\\proj\\cassava-leaf-disease-classification\\train_images\\'):
    if train:
        front = 'train\\'
    else:
        front = 'test\\'
    for fn, label in zip(file_names, labels):
        if os.path.exists(pic_file + front + str(label)):
            shutil.move(pic_file + fn, pic_file + front + str(label))
        else:
            os.makedirs(pic_file + front + str(label))
            shutil.move(pic_file + fn, pic_file + front + str(label))
    print('DONE.')


x_train, x_test, y_train, y_test = train_test_split(train_raw['image_id'], train_raw['label'], test_size=0.2)
xy_train = pd.DataFrame({'x': x_train, 'y': y_train})
xy_test = pd.DataFrame({'x': x_test, 'y': y_train})

file_move(x_train.tolist(), y_train.tolist(), True)
file_move(x_test.tolist(), y_test.tolist(), False)

train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_data_gen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_data_gen.flow_from_directory(
    'D:\\proj\\cassava-leaf-disease-classification\\train_images\\train',
    target_size=(160, 120),
    batch_size=32
)

validation_generator = test_data_gen.flow_from_directory(
    'D:\\proj\\cassava-leaf-disease-classification\\train_images\\test',
    target_size=(160, 120),
    batch_size=32
)

tg = train_data_gen.flow_from_dataframe(dataframe=train_raw,
                                        directory='D:\\proj\\cassava-leaf-disease-classification\\train_images',
                                        subset='training',
                                        x_col='image_id',
                                        y_col='label',
                                        shuffle=True,
                                        target_size=(160, 120),
                                        batch_size=32,
                                        class_mode='categorical')


def tb_callback(exp_name):
    return TensorBoard(log_dir=_log_dir + exp_name, profile_batch=0, histogram_freq=1)


def build_baseline_vgg():
    input_shape = (160, 120, 3)
    baseline_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = baseline_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.25)(x)

    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=baseline_model.input, outputs=predictions)
    model.compile(optimizer=_opt,
                  loss=_loss,
                  metrics=_metrics)
    return model


def build_baseline_model():
    input_shape = (160, 120, 3)
    input = Input(shape=input_shape)
    m = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               activation='tanh',
               kernel_regularizer=l1(0.0001))(input)
    m = MaxPool2D(pool_size=(2, 2))(m)
    m = Dropout(rate=0.1)(m)

    m = Flatten()(m)
    m = Dense(256,
              activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=l1(0.0001))(m)
    m = Dropout(rate=0.12)(m)

    output = Dense(5, activation='softmax')(m)
    m = Model(inputs=input, outputs=output)

    m.compile(optimizer=_opt,
              loss=_loss,
              metrics=_metrics)
    return m


_shuffle = True
_log_dir = './logs/baseline_model/'
_seed = 27
_learning_rate = 0.0001
_schedule = ExponentialDecay(_learning_rate, decay_steps=10_0000, decay_rate=0.96)
_opt = Adam(learning_rate=_schedule)
_es = EarlyStopping(monitor='val_accuracy', patience=20)
_tb = tb_callback('Baseline_model_1')
_callbacks = [_es, _tb]
_metrics = ['accuracy']
_loss = 'categorical_crossentropy'
_epochs = 4

# setup tensorboard, directories
# !rm -rf ./logs
# !mkdir ./logs/
# !mkdir ./logs/baseline_model


baseline_model1 = build_baseline_vgg()
baseline_model1_hist = baseline_model1.fit(train_generator,
                                           epochs=_epochs,
                                           validation_data=validation_generator,
                                           callbacks=_callbacks,
                                           shuffle=_shuffle)

_tb2 = tb_callback('Baseline_model_2')
_callbacks2 = [_es, _tb2]

# setup tensorboard, directories
# !rm -rf ./logs
# !mkdir ./logs/
# !mkdir ./logs/baseline_model


baseline_model2 = build_baseline_model()
baseline_model2_hist = baseline_model2.fit(train_generator,
                                           epochs=_epochs,
                                           validation_data=validation_generator,
                                           callbacks=_callbacks2,
                                           shuffle=_shuffle)

# Splitting the train and test set
x = train_raw['image_id'].tolist()
y = train_raw['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
