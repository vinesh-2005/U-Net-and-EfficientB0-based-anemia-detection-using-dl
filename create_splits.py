import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

MANIFEST = Path("data/processed/manifest.csv")
OUTPUT_DIR = Path("data/splits")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

print("Total patients:", len(df))

# First split:
# 70% train
# 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Split remaining 30% equally:
# 15% validation
# 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

print("\nSaved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "val.csv")
print(OUTPUT_DIR / "test.csv")

print("\nCountry distribution:")
print("\nTrain:")
print(train_df["country"].value_counts())

print("\nValidation:")
print(val_df["country"].value_counts())

print("\nTest:")
print(test_df["country"].value_counts())
