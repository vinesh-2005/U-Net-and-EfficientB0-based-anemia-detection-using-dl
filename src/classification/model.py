"""EfficientNetB0 transfer-learning classifier."""
import tensorflow as tf

def build_classifier(input_size=(224, 224), dropout=0.3, head_units=128, weights="imagenet", trainable_layers=0):
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(*input_size, 3))
    backbone.trainable = trainable_layers > 0
    if trainable_layers: 
        for layer in backbone.layers[:-trainable_layers]:
            layer.trainable = False

    inputs = tf.keras.Input((*input_size, 3), name="input_image")
    
    # Mild data augmentation active only during training
    aug = tf.keras.layers.RandomFlip("horizontal")(inputs)
    aug = tf.keras.layers.RandomRotation(0.03, fill_mode="nearest")(aug)
    aug = tf.keras.layers.RandomZoom(0.04, fill_mode="nearest")(aug)

    x = backbone(aug, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(head_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="efficientnetb0_anemia")

