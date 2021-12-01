import math
import random

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import apply_affine_transform


def transform_images(img, gt, size):
    #  img = tf.image.resize_with_pad(img, 1280, 1280)
    #  gt = tf.image.resize_with_pad(gt, 1280, 1280)

    img, gt = random_scale(img, gt, 0.5, 1.5, 0.25)
    img, gt = random_crop(img, gt, size=size)
    gt = invert_image(gt)
    img, gt = random_rotation(img, gt)
    return img, gt


def random_rotation(image, label):
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image,
                             degree * math.pi / 180,
                             interpolation='nearest')
    label = tfa.image.rotate(label,
                             degree * math.pi / 180,
                             interpolation='nearest')
    return image, label


def invert_image(img):
    return 1 - img


def read_image(file_name, format='L'):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


def random_crop(image, labels, size):
    """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
    combined = tf.concat([image, labels], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
        combined, 0, 0, tf.maximum(size[0], image_shape[0]),
        tf.maximum(size[1], image_shape[1]))
    last_label_dim = tf.shape(labels)[-1]
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.image.random_crop(
        combined_pad,
        size=tf.concat([size, [last_label_dim + last_image_dim]], axis=0))
    img = tf.expand_dims(combined_crop[:, :, :last_image_dim], axis=0)
    label = tf.expand_dims(combined_crop[:, :, last_image_dim:], axis=0)
    return (img, label)


def random_scale(img, gt, min_scale_factor, max_scale_factor, step_size):
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Gets a random scale value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        Raises:
            ValueError: min_scale_factor has unexpected value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor,
                                    num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    f_scale = get_random_scale(min_scale_factor, max_scale_factor, step_size)
    # TODO: cv2 uses align_corner=False
    # TODO: use fvcore (https://github.com/facebookresearch/fvcore/blob/master/fvcore/transforms/transform.py#L377)
    new_size = int(f_scale * 1280)
    image = tf.image.resize(img, [new_size, new_size], 'bilinear')
    groundtruth = tf.image.resize(gt, [new_size, new_size], 'nearest')
    return image, groundtruth
