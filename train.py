import glob
import os
from itertools import cycle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from absl import app, flags, logging
from absl.flags import FLAGS
from model.datagen import random_crop_and_pad_image_and_labels, read_image
from model.loss import HeSho
from model.model import UNetBR
from tensorflow.keras import backend as K
from utils.utils import load_yaml, set_memory_growth

flags.DEFINE_string('cfg_path', './configs/default.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


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

    def generator():
        root = cfg['dataset_path']
        gt_fnames = sorted(glob.glob(os.path.join(root, 'gt', '*.jpeg')))
        img_fnames = sorted(glob.glob(os.path.join(root, 'img', '*.jpeg')))
        iterator = cycle(zip(gt_fnames, img_fnames))
        for gtf, imgf in iterator:
            yield np.array(read_image(imgf)), np.array(read_image(gtf))

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec(shape=(None, None, 1),
                                        dtype=tf.float32),
                          tf.TensorSpec(shape=(None, None, 1),
                                        dtype=tf.float32)))

    dataset = dataset.map(
        lambda img, lab: random_crop_and_pad_image_and_labels(
            img / 255., lab / 255., size=cfg['crop_size']))

    #  dataset = dataset.batch(cfg['batch_size'])

    # define optimizer
    #  steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg['init_lr'],
        decay_steps=cfg['decay_steps'],
        decay_rate=cfg['decay_rate'],
        staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)

    # define network
    model = UNetBR(input_shape=(224, 224, 1))
    model.summary()
    model.compile(loss=HeSho,
                  optimizer=optimizer,
                  metrics=[
                      tfa.metrics.F1Score(num_classes=2,
                                          average="micro",
                                          threshold=0.9),
                      PSNR,
                  ])

    checkpoint_path = "logs/checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(dataset,
              batch_size=2,
              epochs=cfg['epoch'],
              steps_per_epoch=cfg['dataset_len'] // cfg['batch_size'],
              callbacks=[cb_checkpoint])


if __name__ == '__main__':
    app.run(main)
