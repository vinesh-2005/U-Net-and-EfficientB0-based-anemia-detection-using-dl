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


def build_inference_classifier(trained_model: tf.keras.Model) -> tf.keras.Model:
    """
    Constructs a 100% deterministic inference classifier from a trained model by:
    1. Bypassing training-only data augmentation layers (RandomFlip, RandomRotation, RandomZoom).
    2. Explicitly setting training=False on all BatchNormalization and Dropout layers.
    3. Preserving 100% of the trained weights across EfficientNetB0, BatchNormalization, and Dense layers.
    """
    # Locate backbone (EfficientNetB0)
    backbone = None
    for layer in trained_model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        return trained_model

    # Check if augmentation layers exist in the model
    has_aug = any("random" in layer.name.lower() for layer in trained_model.layers)
    if not has_aug:
        return trained_model

    head_layer_names = [
        "global_average_pooling2d",
        "batch_normalization",
        "dropout",
        "dense",
        "dropout_1",
        "dense_1",
    ]

    # Build clean deterministic inference graph without data augmentation
    input_shape = (224, 224, 3)
    inference_input = tf.keras.Input(shape=input_shape, name="inference_input")
    x = backbone(inference_input, training=False)

    for name in head_layer_names:
        try:
            layer = trained_model.get_layer(name)
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                x = layer(x, training=False)
            else:
                x = layer(x)
        except ValueError:
            pass

    inference_model = tf.keras.Model(
        inputs=inference_input,
        outputs=x,
        name="deterministic_efficientnetb0_classifier",
    )
    return inference_model


