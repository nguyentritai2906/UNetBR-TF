import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, backend, layers, models
from tensorflow.keras.layers import Layer


def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


def pad_tensor(target_tensor, tensor):
    # input is CHW
    diffY = target_tensor.shape[2] - tensor.shape[2]
    diffX = target_tensor.shape[3] - tensor.shape[3]
    if diffX == 0 and diffY == 0:
        return tensor

    tensor = tf.pad(
        tensor,
        tf.constant([diffY // 2, diffY - diffY // 2],
                    [diffX // 2, diffX - diffX // 2]))
    return tensor


def double_conv(x, filters, mid_channels=None):
    if not mid_channels:
        mid_channels = filters
    double_conv = keras.Sequential([
        layers.SeparableConv2D(filters=mid_channels,
                               kernel_size=3,
                               padding='same'),
        layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same'),
    ])(x)
    residual = layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(x)
    return tf.add(residual, double_conv)


def down(x, filters):
    maxpool = layers.MaxPool2D()(x)
    return double_conv(maxpool, filters=filters)


def up(x1, x2, filters, bilinear=True, pad_to_size=True):
    if bilinear:
        x1 = layers.UpSampling2D(size=2, interpolation='bilinear')(x1)
    else:
        x1 = layers.Conv2DTranspose(filters // 2, kernel_size=2, stride=2)(x1)

    if pad_to_size:
        x1 = pad_tensor(x2, x1)
    else:
        x2 = crop_tensor(x1, x2)

    x = tf.concat([x2, x1], axis=-1)

    return double_conv(x, filters, filters // 2 if bilinear else None)


def out_conv(x, filters):
    return layers.Conv2D(filters, kernel_size=1)(x)


def unet(x, in_channels, bilinear=True):
    factor = 2 if bilinear else 1

    x1 = double_conv(x, 64)
    x2 = down(x1, 128)
    x2 = layers.Dropout(0.1)(x2)
    x3 = down(x2, 256 // factor)

    x = up(x3, x2, 128 // factor, bilinear)
    x = layers.Dropout(0.1)(x)
    x = up(x, x1, 64, bilinear)
    x = layers.Dropout(0.1)(x)
    logits = out_conv(x, in_channels)
    return logits


def UNetBR(input_tensor=None, input_shape=None, num_block=2, is_train=False):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input
    outputs = []
    for i in range(num_block):
        output = unet(x, 1)
        output = tf.add(x, output)
        output = tf.sigmoid(output)
        outputs.append(output)
        x = output

    if is_train:
        return models.Model(img_input, outputs)
    else:
        return models.Model(img_input, outputs[-1])


# model = UNetBR()
# input_data = np.array(np.random.random_sample((1, 640, 640, 3)),
#                       dtype=np.float32)
# model(input_data)
