
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import VGG16
from functools import partial
from sklearn.model_selection import train_test_split

# Constant Variables:
_auto_tune = tf.data.experimental.AUTOTUNE
_batch_size = 32

_image_width_original = 512
_image_height_original = 512
_image_size = [_image_width_original, _image_height_original]

_image_resize_width = 400
_image_resize_height = 400
_image_resize = [_image_resize_width, _image_resize_height]
print('Model input shape {} x {}.'.format(_image_resize_width, _image_resize_height))

_channels = 3
_n_class = 5
_n_repeat = 4
_img_norm = 255.0

_classes = [str(x) for x in range(_n_class)]
_major_label = 3
_classes_names = ['Cassava Bacterial Blight',
                  'Cassava Brown Streak Disease',
                  'Cassava Green Mottle',
                  'Cassava Mosaic Disease',
                  'Healthy']
_train_file = 'train_tfrecords/'
_train_recs = os.listdir(_train_file)
_test_file = 'test_tfrecords/'
_test_recs = os.listdir(_test_file)
_epochs = 300
_valid_size = 0.1
_train_df = pd.read_csv('train.csv', encoding='utf_8_sig', engine='python')
_file_label_map = dict(zip(_train_df.image_id.tolist(), _train_df.label.astype(int).tolist()))
_random_corp_size = [_image_resize_width, _image_resize_height, _channels]


# Decoding single image:
def decode_img(img,
               n_channels: int = _channels,
               img_size: list = None,
               img_norm : float = _img_norm):
    if img_size is None:
        img_size = _image_size
    img = tf.image.decode_jpeg(img, channels=n_channels)
    img = tf.reshape(img, [*img_size, n_channels])
    img = tf.cast(img, tf.float32) / img_norm
    return img


# Parsing the files
def parse_img(x,
              n_class: int = _n_class):
    feature_description = {'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
                           'target': tf.io.FixedLenFeature([], tf.int64, default_value=-1)}
    parsed_features = tf.io.parse_single_example(x, feature_description)
    img = decode_img(parsed_features['image'])
    label = tf.one_hot(parsed_features['target'], depth=n_class)
    return img, label


# Load data
def load_img(files: list):
    df = tf.data.TFRecordDataset(files)
    df = df.map(parse_img)
    return df


# Resize image
def resize_img(img,
               label,
               shape=None):
    if shape is None:
        shape = _image_resize
    return tf.image.resize(img, shape), label


# Sampling by randomly corp the picture with a fixed size:
def random_corp_sample_img(df,
                           corp_size=None,
                           n_repeat=_n_repeat):
    if corp_size is None:
        corp_size = _image_resize
    df_r = df.map(random_corp_img)
    for _ in range(n_repeat-1):
        df_r = df_r.concatenate(df.map(random_corp_img))
    return df_r


# Random corp:
def random_corp_img(img,
                    label,
                    size=None):
    if size is None:
        size = _random_corp_size
    img = tf.image.random_crop(img, size=size)
    return img, label


# Data Augmentation
def augment_img(img,
                label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 0, 1)
    return img, label


# Train-validation split
_train_fn, _valid_fn = \
    train_test_split(tf.io.gfile.glob(_train_file + 'ld_train*.tfrec'),
                     test_size=_valid_size,
                     random_state=5,
                     shuffle=True)
_test_fn = tf.io.gfile.glob(_test_file + 'ld_test*.tfrec')


# Function for getting the training data set
def get_train_data(train_fn: list = _train_fn,
                   batch_size: int = _batch_size):
    df = load_img(train_fn)

    df_major = df.filter(lambda x, y: tf.argmax(y) == _major_label)
    df_minor = df.filter(lambda x, y: tf.argmax(y) != _major_label)

    df_major = df_major.map(resize_img)
    df_minor = random_corp_sample_img(df_minor)
    df = df_major.concatenate(df_minor)

    df = df.map(augment_img)

    df = df.repeat().shuffle(2048).batch(batch_size)
    return df


# Function for getting the validation data set
def get_valid_data(valid_fn: list = _valid_fn,
                   batch_size: int = _batch_size):
    df = load_img(valid_fn)
    df = df.map(resize_img)
    df = df.batch(batch_size).cache()
    return df


# Function for getting the testing data set
def get_test_data(test_fn: list = _test_fn,
                  batch_size: int = _batch_size):
    df = load_img(test_fn)
    df = df.map(resize_img)
    df = df.batch(batch_size)
    return df


# Reporting the size of training, validation and testing data
def report_data_size(train_f=_train_fn,
                     valid_f=_valid_fn,
                     test_f=_test_fn):
    def count_file(x):
        return sum([int(re.compile(r"-([0-9]*)\.").search(i).group(1)) for i in x])
    n_train, n_valid, n_test = count_file(train_f), count_file(valid_f), count_file(test_f)
    print('Train Images: {} | Validation Images: {} | Test Images: {}'.format(n_train, n_valid, n_test))
    return n_train, n_valid, n_test


train_data = get_train_data()
valid_data = get_valid_data()
test_data = get_test_data()


def tb_callback(exp_name):
    return TensorBoard(log_dir=_log_dir + exp_name, profile_batch=0, histogram_freq=1)


def build_baseline_vgg():
    pass
    return

_n_train, _n_valid, _n_test = report_data_size()
_shuffle = True
_log_dir = './logs/baseline_model/'
_seed = 27
_learning_rate = 0.0001
_schedule = ExponentialDecay(_learning_rate, decay_steps=10_0000, decay_rate=0.96)
_opt = Adam(learning_rate=_schedule)
_es = EarlyStopping(monitor='val_accuracy', patience=20)
_tb = tb_callback('Baseline_model_1')
_callbacks = [_es]
_metrics = ['accuracy']
_loss = 'categorical_crossentropy'
_steps_per_epoch = _n_train // _batch_size

baseline_model1 = build_baseline_vgg()
baseline_model1_hist = baseline_model1.fit(train_data,
                                           epochs=_epochs,
                                           validation_data=valid_data,
                                           steps_per_epoch=_steps_per_epoch,
                                           callbacks=_callbacks,
                                           shuffle=_shuffle)

_tb2 = tb_callback('Baseline_model_2')
_callbacks2 = [_es, _tb2]


