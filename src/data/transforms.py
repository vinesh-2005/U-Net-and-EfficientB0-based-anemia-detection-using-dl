"""TensorFlow image/mask preprocessing and conservative augmentation."""
import tensorflow as tf

def resize_image(image, size): return tf.image.resize(image, size, method="bilinear")
def resize_mask(mask, size): return tf.image.resize(mask, size, method="nearest")
def normalize_rgb(image): return tf.cast(image, tf.float32) / 255.0

def segmentation_augment(image, mask, config):
    image = tf.image.random_brightness(image, config.get("brightness_delta", .08))
    image = tf.image.random_contrast(image, *config.get("contrast_range", [.9, 1.1]))
    return tf.clip_by_value(image, 0, 1), mask

def classifier_augment(image, config):
    image = tf.image.random_brightness(image, config.get("brightness_delta", .05))
    return tf.clip_by_value(image, 0, 1)
