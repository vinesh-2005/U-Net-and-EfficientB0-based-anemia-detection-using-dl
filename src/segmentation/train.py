"""Train and save the U-Net baseline."""
from pathlib import Path
import pandas as pd, tensorflow as tf
from .model import build_unet, bce_dice_loss, dice_coefficient
from .dataset import make_dataset

def train(config: dict, split_csv: str, output_dir: str):
    frame = pd.read_csv(split_csv); size = tuple(config["input_size"]); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model = build_unet(size, config["base_filters"], config.get("dropout", 0))
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss=bce_dice_loss(config["dice_smoothing"]), metrics=[dice_coefficient])
    train_df = frame[(frame.split == "train") & frame.has_valid_mask]; val_df = frame[(frame.split == "val") & frame.has_valid_mask]
    callbacks = [tf.keras.callbacks.ModelCheckpoint(out / "best_unet.keras", monitor="val_loss", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)]
    return model.fit(make_dataset(train_df, size, config["batch_size"], True), validation_data=make_dataset(val_df, size, config["batch_size"]), epochs=config["epochs"], callbacks=callbacks)
