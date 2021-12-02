import os

import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from model.datagen import invert_image, read_image
from model.model import UNetBR

flags.DEFINE_string('image', './images/test_img.jpeg', 'input image')


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    img_path = FLAGS.image
    img = read_image(img_path, 'L')
    img = tf.image.resize_with_pad(img, 1024, 1024)
    img = invert_image(img)

    model = UNetBR(input_shape=img.shape)
    model.load_weights('./model.h5')

    pred = model.predict(tf.expand_dims(img / 255., axis=0))
    pred = invert_image(pred)
    out_path = os.path.join(
        'images/',
        os.path.basename(img_path).split('.')[0] + '_out.jpeg')
    tf.keras.utils.save_img(out_path, tf.squeeze(pred, 0))


if __name__ == '__main__':
    app.run(main)
