import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


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


def random_crop_and_pad_image_and_labels(image, labels, size):
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


if __name__ == "__main__":
    root = './dataset/DIBCO'
    gt_fnames = sorted(glob.glob(os.path.join(root, 'gt', '*.jpeg')))
    img_fnames = sorted(glob.glob(os.path.join(root, 'img', '*.jpeg')))
    iterator = iter(zip(gt_fnames, img_fnames))
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.uint8),
                          tf.TensorSpec(shape=(None, None, 1),
                                        dtype=tf.uint8)))
    dataset = dataset.map(
        lambda img, lab: random_crop_and_pad_image_and_labels(
            img, lab, size=(224, 224)))
    list(dataset.take(1))
