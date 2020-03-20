import os
import time
import yaml
import wandb
import pandas
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
from packaging import version
from datetime import datetime
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.style as style
from edafa import ClassPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models, optimizers
if version.parse(tf.__version__) <= version.parse('2.0'):
    print("TF version:", tf.__version__)
    raise ValueError('This code requires Tensorflow v2.0 or higher')
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from utils import get_random_dcm, macro_f1, macro_f1_loss, print_time, learning_curves, perfomance_grid

# --------------------------------------------------- Main parameters --------------------------------------------------
MODE = 'train'
MODEL_NAME = 'Custom_V1'
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 1 if MODEL_NAME == 'Custom_V1' else 3
BATCH_SIZE = 64
LR = 1e-5
EPOCHS = 100
OPTIMIZER = 'radam'

# ------------------------------------------------ Additional parameters -----------------------------------------------
DATA_PATH = 'data/data.xlsx'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BUFFER_SIZE = 1000
IS_TRAINABLE = False
IS_AUG = False
IS_TTA = False
FILTER_RATE = 5
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
VERBOSE = 2
TEST_MODEL_DIR = 'models/MobileNet_V2_1503_2137'
# TEST_DCM_FILES = ['data/axial/abdomen/2-CHEST  2.0  B30f  SOFT TISSUE-76484/000131.dcm',
#                   'data/axial/chest/3-CHEST  3.0  I70f  2-63964/000015.dcm',
#                   'data/axial/pelvis/4-ABDPEL LUNG ss30-11568/000065.dcm'],
#                   'data/coronal_sagittal/abdomen/601-COR 3MM THICK AVG-77430/000011.dcm',
#                   'data/coronal_sagittal/abdomen_chest/80253-SAG-61052/000011.dcm',
#                   'data/coronal_sagittal/abdomen_pelvis/602-MPR Thick Range-22529/000020.dcm',
#                   'data/coronal_sagittal/chest/701-Display LUNG-25911/000015.dcm',
#                   'data/coronal_sagittal/abdomen_chest_pelvis/701-Coronal-77382/000019.dcm']
TEST_DCM_FILES = [get_random_dcm('data/coronal_sagittal/abdomen/'),
                  get_random_dcm('data/axial/chest/'),
                  get_random_dcm('data/axial/pelvis/'),
                  get_random_dcm('data/coronal_sagittal/abdomen/'),
                  get_random_dcm('data/coronal_sagittal/abdomen_chest/'),
                  get_random_dcm('data/coronal_sagittal/abdomen_pelvis/'),
                  get_random_dcm('data/coronal_sagittal/chest/'),
                  get_random_dcm('data/coronal_sagittal/abdomen_chest_pelvis/')]

"""
    MODE          (str): train, test 
    MODEL_NAME    (str): MobileNet_V2, ResNet_V2, Inception_V3, Inception_ResNet_v2, EfficientNet_B7, Custom_V1
    IMG_HEIGHT    (int): image height 
    IMG_WIDTH     (int): image width
    IMG_CHANNELS  (int): number of image channels
    BATCH_SIZE    (int): number of elements in a batch
    LR          (float): learning rate
    EPOCHS        (int): number of epochs during training
    OPTIMIZER     (str): type of optimizer
    
    BUFFER_SIZE   (int): dataset fills a buffer with BUFFER_SIZE elements
    IS_TRAINABLE (bool): whether to train backbone of transfer learning models
    IS_AUG       (bool): whether to use augmented data
    IS_TTA       (bool): whether to apply test-time augmentation to model
    FILTER_RATE   (int): coefficient affecting the number of model weights
    KERNEL_SIZE (tuple): convolution kernel size
    POOL_SIZE   (tuple): pooling kernel size
    VERBOSE       (int): verbosity mode
"""

# -------------------------------------------- Initialize ArgParse container -------------------------------------------
parser = argparse.ArgumentParser(description='Multi-Label Classification')
parser.add_argument('-mo', '--mode', metavar='', default=MODE, type=str,
                    help='mode: train or test')
parser.add_argument('-mn', '--model_name', metavar='', default=MODEL_NAME, type=str,
                    help='architecture of the model: MobileNet_V2, ResNet_V2, Inception_V3, '
                         'Inception_ResNet_v2, EfficientNet_B7, Custom_V1')
parser.add_argument('-ih', '--img_height', metavar='', default=IMG_HEIGHT, type=int,
                    help='image height')
parser.add_argument('-iw', '--img_width', metavar='', default=IMG_WIDTH, type=int,
                    help='image width')
parser.add_argument('-bas', '--batch_size', metavar='', default=BATCH_SIZE, type=int,
                    help='batch size')
parser.add_argument('-bus', '--buffer_size', metavar='', default=BUFFER_SIZE, type=int,
                    help='buffer size')
parser.add_argument('-lr', '--learning_rate', metavar='', default=LR, type=float,
                    help='learning rate')
parser.add_argument('-ep', '--epochs', metavar='', default=EPOCHS, type=int,
                    help='number of epochs for training')
parser.add_argument('-opt', '--optimizer', metavar='', default=OPTIMIZER, type=str,
                    help='type of an optimizer')
parser.add_argument('-ist', '--is_trainable', action='store_true', default=IS_TRAINABLE,
                    help='whether to train backbone')
parser.add_argument('-isa', '--is_aug', action='store_true', default=IS_AUG,
                    help='whether to use augmented data for training')
parser.add_argument('-tta', '--is_tta', action='store_true', default=IS_TTA,
                    help='whether to use augmented data for training')
parser.add_argument('-fr', '--filter_rate', metavar='', default=FILTER_RATE, type=int,
                    help='coefficient affecting the number of model weights')
parser.add_argument('-ks', '--kernel_size', metavar='', default=KERNEL_SIZE, type=tuple,
                    help='kernel size')
parser.add_argument('-ps', '--pool_size', metavar='', default=POOL_SIZE, type=tuple,
                    help='pool size')
parser.add_argument('-md', '--model_dir', metavar='', default='', type=str,
                    help='model directory for training mode')
parser.add_argument('-ver', '--verbose', metavar='', default=VERBOSE, type=int,
                    help='prediction type')
parser.add_argument('-tmd', '--test_model_dir', metavar='', default=TEST_MODEL_DIR, type=str,
                    help='model directory for testing mode')
parser.add_argument('-tdf', '--test_dcm_files', nargs='+', metavar='', default=TEST_DCM_FILES, type=str,
                    help='list of DICOM files for testing')
parser.add_argument('-ic', '--img_channels', metavar='',
                    default=1 if parser.parse_args().model_name == 'Custom_V1' else 3, type=int, help='image channels')
args = parser.parse_args()

# ------------------------------------------ Data processing and prefetching -------------------------------------------
class DataProcessor():
    def __init__(self):
        pass

    def augment(self, img_src):
        """Function used for augmentation
        Args:
              img_src: original image
        """
        img_aug = tf.identity(img_src)
        img_aug = tf.image.random_flip_left_right(img_aug)
        img_aug = tf.image.random_flip_up_down(img_aug)
        if tf.random.uniform(shape=[1], minval=0, maxval=1) > 0.70:
            img_aug = tf.image.central_crop(img_aug, central_fraction=0.70)
            img_aug = tf.image.resize(img_aug, size=(args.img_height, args.img_width))
        if args.img_channels == 3 and np.random.uniform(low=0.0, high=1.0) > 0.70:
            rotate_deg = np.random.random_integers(low=-25, high=25)
            rotate_rads = rotate_deg * (np.pi / 180)
            img_aug = tfa.image.rotate(img_aug, rotate_rads, interpolation='BILINEAR')
        # Used for debugging
        # self.visualize(img_src, img_aug)
        return img_aug

    def visualize(self, img_src, img_aug):
        """Function used to compare original and augmented image
        Args:
              img_src: original image
              img_aug: augmented image
        """
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Original image')
        plt.imshow(np.squeeze(img_src))
        plt.subplot(1, 2, 2)
        plt.title('Augmented image')
        plt.imshow(np.squeeze(img_aug))
        plt.show()

    def get_filenames_and_labels(self, path_to_data):
        """Function that returns a list of full DICOM filenames and array of labels.
        Args:
            path_to_data: string representing path to xlsx dataset info
              is_shuffle: boolean parameter used for shuffling
        """
        source_df = pandas.read_excel(path_to_data, index_col=None, na_values=['NA'], usecols="B:G")
        path_df = source_df['path']
        label_df = source_df[['axial', 'coronal_sagittal', 'abdomen', 'chest', 'pelvis']]
        paths = list(path_df)
        labels = label_df.to_numpy()
        return paths, labels

    def parse_regular_dcm(self, path, label):
        """Function that returns a tuple of normalized image array and labels array.
        Args:
            filename: string representing path to image
               label: 0/1 one-dimensional array
        """
        img_bytes = tf.io.read_file(path)
        img_input = tfio.image.decode_dicom_image(img_bytes, color_dim=False, dtype=tf.float64, scale='auto')
        img_resized = tf.image.resize(images=img_input, size=(args.img_height, args.img_width))
        if args.model_name == 'Custom_V1':
            img_rgb = tf.identity(img_resized)
        else:
            img_rgb = tf.image.grayscale_to_rgb(img_resized, name=None)
        img_squeezed = tf.squeeze(input=img_rgb, axis=0, name=None)
        img_output = tf.reshape(tensor=img_squeezed, shape=(args.img_height, args.img_width, args.img_channels))
        # Used for debugging
        # self.visualize(img_input, img_squeezed)
        return img_output, label

    def parse_augment_dcm(self, path, label):
        """Function that returns a tuple of augmented normalized image array and labels array.
        Args:
            filename: string representing path to image
               label: 0/1 one-dimensional array
        """
        img_bytes = tf.io.read_file(path)
        img_input = tfio.image.decode_dicom_image(img_bytes, color_dim=False, dtype=tf.float64, scale='auto')
        img_resized = tf.image.resize(images=img_input, size=(args.img_height, args.img_width))
        if args.model_name == 'Custom_V1':
            img_rgb = tf.identity(img_resized)
        else:
            img_rgb = tf.image.grayscale_to_rgb(img_resized, name=None)
        img_squeezed = tf.squeeze(input=img_rgb, axis=0, name=None)
        img_aug = self.augment(img_squeezed)
        img_output = tf.reshape(tensor=img_aug, shape=(args.img_height, args.img_width, args.img_channels))
        # Used for debugging
        # self.visualize(img_input, self.augment(img_squeezed))
        return img_output, label

    def create_dataset(self, filenames, labels, is_aug, is_caching=None, cache_name=None):
        """Load and parse a dataset.
        Args:
             filenames: list of image paths
                labels: numpy array labels
            is_caching: boolean to indicate caching mode
        """
        # Create a first dataset of file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # Used for debugging
        # temp_aug = self.parse_augment_dcm(filenames[11], labels[11])
        # temp = self.parse_regular_dcm(filenames[11], labels[11])

        # Parse and preprocess observations in parallel
        if is_aug:
            dataset = dataset.map(map_func=self.parse_augment_dcm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(map_func=self.parse_regular_dcm, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_caching:
            # This is a small dataset, only load it once, and keep it in memory.
            # use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
            if isinstance(cache_name, str):
                cache_path = os.path.join('data', cache_name)
                if not os.path.exists(os.path.split(cache_path)[0]):
                    os.makedirs(os.path.split(cache_path)[0])
                dataset = dataset.cache(cache_path)
            else:
                dataset = dataset.cache()

        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=args.buffer_size)     # len(filenames)

        # Repeats the dataset so each original value is seen `count` times
        dataset = dataset.repeat(count=1)

        # Batch the data for multiple steps
        dataset = dataset.batch(batch_size=args.batch_size)

        # Fetch batches in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

# ------------------------------ Custom callback to log metrics of each batch individually -----------------------------
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_loss = []
        self.batch_acc = []
        self.batch_lr = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_loss.append(logs['loss'])
        self.batch_acc.append(logs['macro_f1'])
        self.model.reset_metrics()

    def on_train_batch_begin(self, batch, logs=None):
        if self.model.optimizer.__class__.__name__ == 'SGD':
            clr = tf.keras.backend.get_value(self.model.optimizer.lr)
            lr = tf.keras.backend.get_value(clr.__call__(batch))
        else:
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)

        self.batch_lr.append(lr)
        if batch > 0:
            print(" - lr: {:.6f}".format(lr))
        self.model.reset_metrics()

# -------------------------------------- Inherit predictor class for TTA testing ---------------------------------------
class TTAPredictor(ClassPredictor):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_patches(self, patches):
        return self.model.predict(patches)

# --------------------------------------------------- Neural network ---------------------------------------------------
class Net:
    def __init__(self):
        pass

    def get_optimizer(self, optimizer, learning_rate):
        if optimizer == 'adam':
            opt = optimizers.Adam(lr=learning_rate)
        elif optimizer == 'adamax':
            opt = optimizers.Adamax(lr=learning_rate)
        elif optimizer == 'radam':
            opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            # lambda x: 1.
            # lambda x: gamma ** x
            # lambda x: 1 / (2.0 ** (x - 1))
            lr_schedule = tfa.optimizers.cyclical_learning_rate.CyclicalLearningRate(initial_learning_rate=learning_rate,
                                                                                     maximal_learning_rate=100*learning_rate,
                                                                                     step_size=25,
                                                                                     scale_mode="iterations",
                                                                                     scale_fn=lambda x: 0.95 ** x,
                                                                                     name="CustomScheduler")
            opt = optimizers.SGD(learning_rate=lr_schedule)
        else:
            raise ValueError('Undefined OPTIMIZER_TYPE!')
        return opt

    def save_model(self, model):
        print('-' * 56)
        print('Saving of the model architecture...')
        start = time.time()
        with open(os.path.join(args.model_dir, 'architecture.json'), 'w') as f:
            f.write(model.to_json())
        end = time.time()
        print('Saving of the model architecture takes ({:1.3f} seconds)'.format(end - start))
        print('-' * 56)

    def load_model(self, model_dir):
        print('-' * 59)
        print('Loading the model and its weights...')
        start = time.time()
        architecture_path = os.path.join(model_dir, 'architecture.json')
        weights_path = os.path.join(model_dir, 'weights.h5')
        with open(architecture_path, 'r') as f:
            model = models.model_from_json(f.read(), custom_objects={'KerasLayer': hub.KerasLayer})
        model.load_weights(weights_path)
        # Used for loading the whole model
        # model = tf.keras.models.load_model(filepath='model.h5',
        #                                    custom_objects={'KerasLayer': hub.KerasLayer},
        #                                    compile=False)
        # Old verson of model compilation (not suitable for MODE = 'test')
        # model.compile(optimizer=self.get_optimizer(learning_rate=self.config.learning_rate),
        #               loss=macro_f1_loss,
        #               metrics=[macro_f1])
        # Optional case for retraining the model
        # model.compile(optimizer=self.get_optimizer(optimizer=optimizer, learning_rate=learning_rate),
        #               loss=macro_f1_loss,
        #               metrics=[macro_f1])
        end = time.time()
        print('Loading the model and its weights takes ({:1.3f} seconds)'.format(end - start))
        print('-' * 59)
        return model

    def get_custom_model(self):

        # -------------------------------------------------- 224, 224 --------------------------------------------------
        input = layers.Input(shape=(args.img_height, args.img_width, args.img_channels))

        # -------------------------------------------------- 224, 224 --------------------------------------------------
        block_1 = self.get_conv_block(input_layer=input, level=1)

        # -------------------------------------------------- 112, 112 --------------------------------------------------
        block_2 = self.get_conv_block(input_layer=block_1, level=2)
        down_2_1 = self.get_downsampling_block(input_layer=block_1, end_level=2, start_level=1)
        concat_2 = layers.concatenate([block_2, down_2_1])

        # --------------------------------------------------- 56, 56 ---------------------------------------------------
        block_3 = self.get_conv_block(input_layer=concat_2, level=3)
        down_3_1 = self.get_downsampling_block(input_layer=block_1, end_level=3, start_level=1)
        down_3_2 = self.get_downsampling_block(input_layer=block_2, end_level=3, start_level=2)
        concat_3 = layers.concatenate([block_3, down_3_1, down_3_2])

        # --------------------------------------------------- 28, 28 ---------------------------------------------------
        block_4 = self.get_conv_block(input_layer=concat_3, level=4)
        down_4_1 = self.get_downsampling_block(input_layer=block_1, end_level=4, start_level=1)
        down_4_2 = self.get_downsampling_block(input_layer=block_2, end_level=4, start_level=2)
        down_4_3 = self.get_downsampling_block(input_layer=block_3, end_level=4, start_level=3)
        concat_4 = layers.concatenate([block_4, down_4_1, down_4_2, down_4_3])

        # --------------------------------------------------- 14, 14 ---------------------------------------------------
        block_5 = self.get_conv_block(input_layer=concat_4, level=5)
        down_5_1 = self.get_downsampling_block(input_layer=block_1, end_level=5, start_level=1)
        down_5_2 = self.get_downsampling_block(input_layer=block_2, end_level=5, start_level=2)
        down_5_3 = self.get_downsampling_block(input_layer=block_3, end_level=5, start_level=3)
        down_5_4 = self.get_downsampling_block(input_layer=block_4, end_level=5, start_level=4)
        concat_5 = layers.concatenate([block_5, down_5_1, down_5_2, down_5_3, down_5_4])

        output = layers.GlobalAveragePooling2D()(concat_5)
        model_layer = models.Model(inputs=input, outputs=output, name='backbone')
        return model_layer

    def get_downsampling_block(self, input_layer, end_level, start_level):
        stride = 2**(end_level - start_level)
        output = layers.AveragePooling2D(pool_size=args.pool_size,
                                         strides=np.dot(stride, (1, 1)),
                                         padding="same")(input_layer)
        return output

    def get_conv_block(self, input_layer, level):

        layer_1_1 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(5, 5),
                                  dilation_rate=(2, 2),
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tfa.activations.gelu)(input_layer)
        layer_1_2 = layers.MaxPool2D(pool_size=args.pool_size, padding="same")(layer_1_1)

        layer_2_1 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tfa.activations.gelu)(input_layer)
        layer_2_2 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same",
                                  activation=tfa.activations.gelu)(layer_2_1)

        layer_3_1 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tfa.activations.gelu)(input_layer)
        layer_3_2 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same",
                                  activation=tfa.activations.gelu)(layer_3_1)

        layer_4_1 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tfa.activations.gelu)(input_layer)
        layer_4_2 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="same",
                                  activation=tfa.activations.gelu)(layer_4_1)
        layer_4_3 = layers.Conv2D(filters=args.filter_rate * 2 ** level,
                                  kernel_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same",
                                  activation=tfa.activations.gelu)(layer_4_2)

        concat = layers.concatenate([layer_1_2, layer_2_2, layer_3_2, layer_4_3])
        output = tfa.layers.GroupNormalization(groups=2 ** level, axis=3)(concat)
        # output = layers.Dropout(rate=0.05*(level+1))(norm)
        return output

    def get_model(self):
        if args.model_name == 'MobileNet_V2':
            model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
        elif args.model_name == 'ResNet_V2':
            model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
        elif args.model_name == 'Inception_V3':
            model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
        elif args.model_name == 'Inception_ResNet_v2':
            model_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
        elif args.model_name == 'EfficientNet_B7':
            model_url = 'https://tfhub.dev/google/efficientnet/b7/feature-vector/1'
        elif args.model_name == 'Custom_V1':
            pass
        else:
            raise ValueError('Incorrect MODEL_TYPE value!')

        if args.model_name == 'Custom_V1':
            model_layer = self.get_custom_model()
        else:
            model_layer = hub.KerasLayer(model_url, input_shape=(args.img_height,
                                                                 args.img_width,
                                                                 args.img_channels))
            model_layer.trainable = args.is_trainable


        model = tf.keras.Sequential([model_layer,
                                     layers.Dense(1024, activation='relu', name='hidden_layer'),
                                     layers.Dense(5, activation='sigmoid', name='output')],
                                    name=args.model_name)

        model.compile(optimizer=self.get_optimizer(optimizer=args.optimizer,
                                                   learning_rate=args.learning_rate),
                      loss=macro_f1_loss,
                      metrics=[macro_f1])
        return model

    def train_model(self):
        # -------------------------------------- Data processing and prefetching ---------------------------------------
        data_processor = DataProcessor()
        paths, labels = data_processor.get_filenames_and_labels(path_to_data=DATA_PATH)
        # Comment for training/uncomment for debugging
        # paths = paths[0:4000]
        # labels = labels[0:4000]

        X_train, X_val, y_train, y_val = train_test_split(paths, labels, shuffle=True, test_size=0.2, random_state=11)
        print('-'*59)
        print("Number of DICOM files for training.....: {} ({:.1%})".format(len(X_train),
                                                                            round(len(X_train)/(len(X_train)+len(X_val)), 1)))
        print("Number of DICOM files for validation...: {} ({:.1%})".format(len(X_val),
                                                                            round(len(X_val)/(len(X_train)+len(X_val)), 1)))
        print('-'*59)

        suffix = '_aug' if args.is_aug else ''
        suffix = suffix + '_Custom_V1' if args.model_name == 'Custom_V1' else suffix
        train_ds = data_processor.create_dataset(X_train, y_train, is_aug=args.is_aug, is_caching=True,
                                                 cache_name='train' + suffix)
        val_ds = data_processor.create_dataset(X_val, y_val, is_aug=False, is_caching=True,
                                               cache_name='val' + suffix)

        # Comment for training/uncomment for debugging
        # train_ds = data_processor.create_dataset(X_train, y_train, is_aug=args.is_aug, is_caching=True, cache_name=None)
        # val_ds = data_processor.create_dataset(X_val, y_val, is_aug=False, is_caching=True, cache_name=None)

        for file, label in train_ds.take(1):
            print('-' * 59)
            print("Shape of features array................: {}".format(file.numpy().shape))
            print("Shape of labels array..................: {}".format(label.numpy().shape))
            print('-'*59)

        # ------------------------------------------  Initialize W&B project -------------------------------------------
        run_time = datetime.now().strftime("%d%m_%H%M")
        run_name = args.model_name + '_{}'.format(run_time)
        args.model_dir = os.path.join('models', run_name)
        os.makedirs(args.model_dir)

        params = dict(img_size=(args.img_height, args.img_width, args.img_channels),
                      model_name=args.model_name,
                      model_dir=args.model_dir,
                      optimizer=args.optimizer,
                      filter_rate=args.filter_rate,
                      kernel_size=args.kernel_size,
                      pool_size=args.pool_size,
                      batch_size=args.batch_size,
                      buffer_size=args.buffer_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      pretrained=False if args.model_name == 'Custom_V1' else True,
                      trainable_backbone=args.is_trainable)
        wandb.init(project="mlc", dir=args.model_dir, name=run_name, sync_tensorboard=True, config=params)
        wandb.run.id = wandb.run.id
        config = wandb.config

        # -------------------------------------------  Initialize callbacks --------------------------------------------
        batch_stats = CollectBatchStats()
        csv_logger = CSVLogger(os.path.join(args.model_dir, 'logs.csv'), separator=',', append=False)
        check_pointer = ModelCheckpoint(filepath=os.path.join(args.model_dir, 'weights.h5'),
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='min',
                                        verbose=1)
        wb_logger = wandb.keras.WandbCallback(monitor='val_loss',
                                              mode='min',
                                              save_weights_only=False,
                                              save_model=False,
                                              log_evaluation=False,
                                              verbose=1)
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.005,
                                  patience=5,
                                  mode='min',
                                  verbose=1)

        # ------------------------------------------- Show training options --------------------------------------------
        print('-' * 59)
        print('Training options:')
        print('Model name...........: {}'.format(args.model_name))
        print('Apply augmentation...: {}'.format(args.is_aug))
        print('Trainable backbone...: {}'.format(args.is_trainable))
        print('Model directory......: {}'.format(args.model_dir))
        print('Image dimensions.....: {}x{}x{}'.format(args.img_height, args.img_width, args.img_channels))
        print('Epochs...............: {}'.format(args.epochs))
        print('Batch size...........: {}'.format(args.batch_size))
        print('Buffer size..........: {}'.format(args.buffer_size))
        print('Kernel size..........: {}'.format(args.kernel_size))
        print('Pool size............: {}'.format(args.pool_size))
        print('Filter rate..........: {}'.format(args.filter_rate))
        print('Optimizer............: {}'.format(args.optimizer))
        print('Learning rate........: {}'.format(args.learning_rate))
        print('-' * 59)

        # --------------------------------------- Get model and then train it ------------------------------------------
        model = self.get_model()
        self.save_model(model=model)
        model.summary()

        # Check model's operability
        for batch in val_ds:
            print('-' * 80)
            print("Model operability test: {}".format(np.squeeze(model.predict(batch)[:1])))
            print('-' * 80)
            break

        start = time.time()
        history = model.fit(x=train_ds,
                            epochs=args.epochs,
                            validation_data=val_ds,
                            callbacks=[csv_logger,
                                       wb_logger,
                                       check_pointer,
                                       batch_stats,
                                       earlystop])
        end = time.time()
        print('\nTraining of the model took: {}'.format(print_time(end-start)))

        # ---------------------------------- Get additional metrics and visualization ----------------------------------
        batch_logs = pandas.DataFrame({'Loss': batch_stats.batch_loss, 'Macro F1': batch_stats.batch_acc,
                                       'LR': batch_stats.batch_lr})
        batch_logs = batch_logs[['Loss', 'Macro F1', 'LR']]
        batch_logs_name = os.path.join(args.model_dir, 'batch_logs.xlsx')
        batch_logs.to_excel(batch_logs_name, sheet_name='Logs', index=True, startrow=0, startcol=0)

        curves_save_path = os.path.join(args.model_dir, 'loss vs macro F1 (' + args.model_name + ').png')
        losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history=history, fname=curves_save_path)

        print('-' * 54)
        print("Loss.............: {:.2f}".format(val_losses[-1]))
        print("Macro F1-score...: {:.2f}".format(val_macro_f1s[-1]))
        print('-' * 54)

        # --------------------- Performance table of the model with different levels of threshold ----------------------
        label_names = ['axial', 'coronal_sagittal', 'abdomen', 'chest', 'pelvis']
        perfomance_grid(ds=val_ds, target=y_val, label_names=label_names, model=model, save_dir=args.model_dir)

    def test_model(self, dcm_paths, thresh=0.5):

        # -------------------------------------------- Show testing options --------------------------------------------
        print('-' * 59)
        print('Testing options:')
        print('Test model name......: {}'.format(args.test_model_dir))
        print('Trainable backbone...: {}'.format(args.test_dcm_files))
        print('Application of TTA...: {}'.format(args.is_tta))
        print('Verbosity mode.......: {}'.format(args.verbose))
        print('-' * 59)

        # -------------------------------------- Getting of a YAML configuration ---------------------------------------
        for root, dirs, files in os.walk(args.test_model_dir):
            for file in files:
                if file.endswith(".yaml"):
                    config_path = os.path.join(root, file)

        if 'config_path' in globals() or 'config_path' in locals():
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError('There is no a YAML config file!')

        # --------------------------------- Model loading and getting of the prediction --------------------------------
        model = self.load_model(model_dir=args.test_model_dir)

        for dcm_path in dcm_paths:
            # Read and preprocess DICOM
            img_bytes = tf.io.read_file(dcm_path)
            source_img = tfio.image.decode_dicom_image(img_bytes, color_dim=False, dtype=tf.float64, scale='auto')
            resized_img = tf.image.resize(images=source_img, size=(config['img_size']['value'][0],
                                                                   config['img_size']['value'][1]))
            out_img = tf.image.grayscale_to_rgb(resized_img, name=None)

            # Generate prediction and process probabilities and labels
            classes = ['axial', 'coronal_sagittal', 'abdomen', 'chest', 'pelvis']
            mlb = MultiLabelBinarizer(classes=classes)
            mlb.fit(y=classes)
            start_reg = time.time()
            model_probs = model.predict(out_img)
            end_reg = time.time()
            predict_probs, predict_labels = self.process_probs(mlb, model_probs, thresh)

            # Generate prediction based on TTA and then process probabilities and labels
            if args.is_tta:
                tta_config = '{"augs":["NO","FLIP_LR", "FLIP_UD", "BRIGHT"], "mean":"ARITH", "bits":8}'
                tta_predictor = TTAPredictor(model, tta_config)
                out_img_tta = tf.keras.backend.get_value(out_img)
                start_tta = time.time()
                model_probs_tta = tta_predictor.predict_images(out_img_tta)
                end_tta = time.time()
                predict_probs_tta, predict_labels_tta = self.process_probs(mlb, model_probs_tta, thresh)

            # Show DICOM, source label (if exists) and predicted label
            if args.verbose == 0:
                print('-' * 100)
                print('DICOM file.................: {}'.format(dcm_path))
                print('Organs threshold...........: {}'.format(thresh))
                print('Predicted labels...........: {}'.format(predict_labels))
                print('Predicted labels(TTA)......: {}'.format(predict_labels_tta)) if args.is_tta else ''
                print('Label probabilities........: {}'.format(predict_probs))
                print('Label probabilities(TTA)...: {}'.format(predict_probs_tta)) if args.is_tta else ''
                print('Prediction time............: {:1.2f} seconds'.format(end_reg - start_reg))
                print('Prediction time(TTA).......: {:1.2f} seconds'.format(end_tta - start_tta)) if args.is_tta else ''
                print('-' * 100)
            elif args.verbose == 1:
                style.use('default')
                plt.figure(dcm_path, figsize=(6, 9))
                plt.imshow(np.squeeze(source_img.numpy()), cmap='gray')
                if args.is_tta:
                    plt.title('\n\nPredicted labels: {}\n\nPredicted labels (TTA): {}'
                              '\n\nLabel probabilities: {}\n\nLabel probabilities (TTA): {}'
                              .format(predict_labels, predict_labels_tta,
                                      predict_probs, predict_probs_tta),
                              fontsize=12)
                else:
                    plt.title('\n\nPredicted labels: {}''\n\nLabel probabilities: {}\n'
                              .format(predict_labels, predict_probs), fontsize=12)
                plt.show()
            elif args.verbose == 2:
                view = dcm_path.split(os.path.sep)[1]
                organ = dcm_path.split(os.path.sep)[2].split('_')
                source_labels = organ
                source_labels.insert(0, view)

                style.use('default')
                plt.figure(dcm_path, figsize=(6, 9))
                plt.imshow(np.squeeze(source_img.numpy()), cmap='gray')
                if args.is_tta:
                    plt.title('\n\nSource labels: {}'
                              '\n\nPredicted labels: {}\n\nPredicted labels (TTA): {}'
                              '\n\nLabel probabilities: {}\n\nLabel probabilities (TTA): {}'
                              .format(source_labels,
                                      predict_labels, predict_labels_tta,
                                      predict_probs, predict_probs_tta),
                              fontsize=12)
                else:
                    plt.title('\n\nSource labels: {}\n\nPredicted labels: {}\n\nLabel probabilities: {}'
                              .format(source_labels, predict_labels, predict_probs), fontsize=12)
                plt.show()
            else:
                raise ValueError('Incorrect VERBOSE value!')

    def process_probs(self, mlb, model_probs, thresh):
        predict_labels_bin = np.zeros([1, 5], dtype=int)
        predict_probs = []
        # Processing of the view labels
        if model_probs[0, 0] > model_probs[0, 1]:
            predict_labels_bin[0, 0] = 1
            predict_probs.append(model_probs[0, 0])
        else:
            predict_labels_bin[0, 1] = 1
            predict_probs.append(model_probs[0, 1])

        # Processing of the labels belonging to axial view
        if model_probs[0, 0] > model_probs[0, 1]:
            max_val = np.amax(model_probs[0, 2:5])
            max_idx = np.where(model_probs[0, 2:5] == max_val)
            max_idx = int(max_idx[0])
            if max_val > thresh:
                predict_probs.append(max_val)
                predict_labels_bin[0, max_idx+2] = 1
        # Processing of the labels belonging coronal & sagittal view
        else:
            for idx in range(2, 5):
                if model_probs[0, idx] > thresh:
                    predict_labels_bin[0, idx] = 1
                    predict_probs.append(model_probs[0, idx])
        predict_probs = ['%.2f' % elem for elem in predict_probs]
        predict_labels = mlb.inverse_transform(yt=predict_labels_bin)
        predict_labels = list(predict_labels[0])
        return predict_probs, predict_labels

# ------------------------------------------------------- Handler ------------------------------------------------------
if __name__ == '__main__':
    net = Net()
    if args.mode == 'train':
        net.train_model()
    elif args.mode == 'test':
        net.test_model(dcm_paths=args.test_dcm_files)
    else:
        raise ValueError('Incorrect MODE value!')

    print('-' * 30)
    print(args.mode.capitalize() + 'ing is finished!')
    print('-' * 30)
