"""Conventional configurable U-Net with BCE + Dice loss."""
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)
    return tf.reduce_mean((2 * tf.reduce_sum(y_true * y_pred, axes) + smooth) /
                          (tf.reduce_sum(y_true, axes) + tf.reduce_sum(y_pred, axes) + smooth))

def bce_dice_loss(smooth=1.0):
    bce = tf.keras.losses.BinaryCrossentropy()
    def loss(y_true, y_pred): return bce(y_true, y_pred) + (1.0 - dice_coefficient(y_true, y_pred, smooth))
    return loss

def _block(x, filters, name, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x); x = tf.keras.layers.Activation("relu")(x)
    return tf.keras.layers.Dropout(dropout)(x) if dropout else x

def build_unet(input_size=(256, 256), base_filters=32, dropout=0.0):
    inputs = tf.keras.Input((*input_size, 3)); skips, x = [], inputs
    for i in range(4):
        x = _block(x, base_filters * 2**i, f"enc{i}", dropout); skips.append(x); x = tf.keras.layers.MaxPooling2D()(x)
    x = _block(x, base_filters * 16, "bottleneck", dropout)
    for i in reversed(range(4)):
        filters = base_filters * 2**i; x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skips[i]]); x = _block(x, filters, f"dec{i}", dropout)
    return tf.keras.Model(inputs, tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x), name="unet_conjunctiva")
