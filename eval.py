import glob
import os
from itertools import cycle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from model.datagen import random_crop, read_image
from model.loss import HeSho
from model.model_func import UNetBR
from utils.utils import ProgressBar, load_yaml, set_memory_growth

flags.DEFINE_string('cfg_path', './configs/default.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


class SaveBestModel(Callback):
    def __init__(self, save_best_metric='f1_score', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log(
        (max_pixel**2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    file_reader = tf.io.read_file('./test_img.jpeg', 'file_reader')
    image_reader = tf.image.decode_jpeg(file_reader,
                                        channels=1,
                                        name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)

    model = UNetBR(dims_expander)

    # load checkpoint
    checkpoint_dir = './logs/checkpoint'
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    pred = model.predict(dims_expander)
    tf.keras.utils.save_img('test_img_out.jpeg', tf.squeeze(pred, 0))


if __name__ == '__main__':
    app.run(main)
