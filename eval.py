import os

import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.keras import backend as K

from model.datagen import read_image
from model.model import UNetBR
from utils.utils import load_yaml

flags.DEFINE_string('cfg_path', './configs/default.yaml', 'config file path')


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    img = read_image('./images/test_img.jpeg', 'L')

    model = UNetBR(input_shape=img.shape)
    model.load_weights('./model.h5')

    pred = model.predict(tf.expand_dims(img / 255., axis=0))
    tf.keras.utils.save_img('images/test_img_out.jpeg', tf.squeeze(pred, 0))


if __name__ == '__main__':
    app.run(main)
